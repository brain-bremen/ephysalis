"""The MC_Rack reader (ephysalis.legacy.mcd) on synthetic files: header, samples, digital lines,
trigger events with trial synchronisation, a recording split over two files, the index with and
without trigger events, and the pipeline (trials of VStim 1.x through vstim_td01)."""

import struct
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytest

from ephysalis.cli.main import main
from ephysalis.io.base import open_session
from ephysalis.legacy.mcd import EPOCH, McdRecordingExtractor, group_files, parse_streams, read_index, scan_file
from ephysalis.session import StreamKind

PER_BLOCK = 2500  # 100 ms at 25 kHz
RESOLUTION = 1.25e-7  # V per count


def _chunk(tag: bytes, body: bytes) -> bytes:
    return tag.ljust(8) + struct.pack("<q", len(body)) + body


def _stream_header(type_name: str, tag: str, rate: float, channels: list[tuple[int, int, str]], record: int = 1104) -> bytes:
    head = bytearray(804)
    head[0:2] = struct.pack("<H", 0)
    head[2:2 + len(type_name)] = type_name.encode()
    struct.pack_into("<IH", head, 784, int(rate * 1000), len(channels))
    head[792:800] = tag.encode()
    for index, hwid, label in channels:
        rec = bytearray(record)
        struct.pack_into("<II", rec, 4, index, hwid)
        rec[790:790 + len(label)] = label.encode()
        head += rec
    return bytes(head)


def _stream_format(unit: str, resolution: float, per_block: int) -> bytes:
    body = bytearray(24)
    body[2] = ord(unit) if unit else 0x20
    struct.pack_into("<HHd", body, 4, 16, 0x8000, resolution)
    struct.pack_into("<HH", body, 16, 2, per_block)
    return bytes(body)


def write_mcd(path, start: datetime, elec: np.ndarray, digi: np.ndarray, triggers=(), index_every: int = 2) -> None:
    """MC_Rack 4.0 file: ``trig0001`` (events [(sample, trial, stimulus)]), ``digi0001`` (16-bit
    word per sample), ``elec0001`` (counts, samples x channels); blocks of 100 ms."""
    n_channels = elec.shape[1]
    delta = start - EPOCH
    t0 = (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
    streams = [("trigger", "trig0001", [(0, 252, "61D")], "V", RESOLUTION),
               ("digital", "digi0001", [(0, 256, "D1")], "", 0.0),
               ("analog", "elec0001", [(k, k, f"{k + 1:02d}A") for k in range(n_channels)], "V", RESOLUTION)]
    header = _chunk(b"MCSHDR", struct.pack("<6H", 11, 0, 0, 4, 0, 5).ljust(2320, b"\0"))
    for type_name, tag, channels, unit, res in streams:
        header += _chunk(b"STRMHDR", _stream_header(type_name, tag, 25000.0, channels))
        header += _chunk(b"STRMFMT", _stream_format(unit, res, PER_BLOCK))
    out = bytearray(_chunk(b"MCSSTRM", b"")[:8] + struct.pack("<q", -1) + _chunk(b"LISThdr", header))
    position_field = len(out) + 16
    out += _chunk(b"IDX1POS", struct.pack("<q", 0))
    groups = []
    for b in range(elec.shape[0] // PER_BLOCK):
        lo, hi = b * PER_BLOCK, (b + 1) * PER_BLOCK
        times = struct.pack("<QQ", t0 + b * 100_000_000, t0 + (b + 1) * 100_000_000)
        if b % index_every == 0:
            groups.append((t0 + b * 100_000_000, len(out)))
        events = b"".join(struct.pack("<QHHHii", t0 + int(s * 40_000), 0xFFFF, 0, 0, trial, stim)
                          for s, trial, stim in triggers if lo <= s < hi)
        out += _chunk(b"trig0001", times + events)
        out += _chunk(b"digi0001", times + digi[lo:hi].astype("<u2").tobytes())
        out += _chunk(b"elec0001", times + (elec[lo:hi].astype(np.int32) + 0x8000).astype("<u2").tobytes())
    struct.pack_into("<q", out, position_field, len(out))
    index = b"".join(struct.pack("<Qq", t, p) for t, p in groups)
    out += _chunk(b"IDX1MCS", b"\x01\x00" + b"\0" * 301) + _chunk(b"IDXIDX", index)
    out += _chunk(b"IDXSWEEP", struct.pack("<QQ", t0, t0 + elec.shape[0] * 40_000))
    path.write_bytes(bytes(out))


def _signals(n, n_channels=4, seed=0):
    rng = np.random.default_rng(seed)
    elec = rng.integers(-3000, 3000, (n, n_channels)).astype(np.int16)
    digi = np.zeros(n, np.uint16)
    return elec, digi


def test_mcd_reader(tmp_path):
    start = datetime(2011, 12, 19, 12, 25, 23, 578125)
    elec, digi = _signals(25_000)  # 1 s = 10 blocks
    digi[1000:1300] |= 1  # line 1 pulse
    digi[5000:] |= 1 << 3  # line 4 rises, stays up
    triggers = [(1010, 114, 531), (6005, 115, 515)]
    session = tmp_path / "20111219"
    session.mkdir()
    write_mcd(session / "20111219_dotmap.mcd", start, elec, digi, triggers)
    index = read_index(session / "20111219_dotmap.mcd")
    assert index.start == start and len(index.group_times) == 10 and index.block_ns == 100_000_000
    walked = scan_file(session / "20111219_dotmap.mcd")
    assert all((walked.offsets[t] == o).all() for t, o in index.offsets.items())
    assert [s.tag for s in index.streams] == ["trig0001", "digi0001", "elec0001"]
    assert index.events["trial"].tolist() == [114, 115] and index.events["stimulus"].tolist() == [531, 515]

    s = open_session(session)
    assert s.format.name == "mcd" and len(s.runs) == 1
    run = s.runs[0]
    assert run.aux_stream == "digi0001" and run.streams == ["digi0001", "elec0001"] and run.start_time == start
    rec = McdRecordingExtractor([str(session / "20111219_dotmap.mcd")], "elec0001")
    np.testing.assert_array_equal(rec.get_traces(start_frame=1234, end_frame=7777), elec[1234:7777])
    assert list(rec.channel_ids) == ["01A", "02A", "03A", "04A"]
    assert rec.get_channel_gains()[0] == pytest.approx(0.125)  # µV per count
    ttl = run.format.read_ttl_events(run)
    assert ttl.values.tolist() == [[1000, 1, 1], [1300, 1, 0], [5000, 4, 1]]
    td = run.format.read_trial_descriptors(run)
    assert td.values.tolist() == [[1010, 114, 531], [6005, 115, 515]]


def test_split_recording_and_pipeline(tmp_path):
    """A recording over two files is one run; a later recording another; trials from the
    trigger events (TD) and the .tdr; DH5 export."""
    from test_vstim_td01 import write_tdr

    session = tmp_path / "20101118"
    session.mkdir()
    start = datetime(2010, 11, 18, 10, 0, 0)
    elec, digi = _signals(50_000, seed=1)
    for s in (2000, 27_000):  # trial start pulses on line 1
        digi[s:s + 500] |= 1
    write_mcd(session / "20101118_task.mcd", start, elec[:25_000], digi[:25_000], [(2005, 7, 3)], index_every=1)
    write_mcd(session / "20101118_task0001.mcd", start.replace(second=1), elec[25_000:], digi[25_000:],
              [(2005, 8, 4)], index_every=3)
    write_mcd(session / "20101118_task0002.mcd", start.replace(minute=5), *_signals(12_500, seed=2))
    groups = group_files(sorted(session.glob("*.mcd")))
    assert [[p.name for p in g] for g in groups] == [["20101118_task.mcd", "20101118_task0001.mcd"],
                                                     ["20101118_task0002.mcd"]]
    s = open_session(session)
    run = s.runs[0]
    assert [r.name for r in s.runs] == ["20101118_task", "20101118_task0002"] and run.duration_s == 2.0
    rec = run.get_recording(StreamKind.AUX)
    assert rec.get_num_samples() == 50_000
    np.testing.assert_array_equal(run.format.get_recording(run, StreamKind.AUX).get_traces()[:, 0], digi)
    assert run.format.read_trial_descriptors(run).values.tolist() == [[2005, 7, 3], [27_005, 8, 4]]
    write_tdr(session / "2010_11_18.tdr", [(7, 3, 1, 1.0), (8, 4, 7, 1.0)], day="18.11.2010")
    (tmp_path / "session.toml").write_text(
        'schema_version = 5\nsession_id = "20101118"\nsubject_id = "iris"\n'
        '[[runs]]\nname = "20101118_task0002"\ninclude = false\n'
        '[probes.v1]\nstream = "elec0001"\nchannels = ["01A", "02A", "03A"]\n'
        '[aux_channels]\nanalog = {}\n[aux_channels.digital]\ntrial_start = 0\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["timing", str(session), *common]) == 0
    trials = pd.read_csv(next((tmp_path / "work" / "trials").glob("*.csv")))
    assert trials.vstim_trial.tolist() == [7, 8] and trials.condition.tolist() == [3, 4]
    assert trials.outcome.tolist() == [1, 7] and trials.start_time.tolist() == pytest.approx([0.08, 1.08])
    assert main(["lfp", str(session), *common, "--n-jobs", "1"]) == 0
    assert main(["export", str(session), *common, "--output-format", "dh5", "--raw", "-o", str(tmp_path / "o.dh5")]) == 0
    with h5py.File(tmp_path / "o.dh5") as f:
        np.testing.assert_array_equal(f["CONT1"]["DATA"][:, 0], elec[:, 0])
        assert f["CONT1"].attrs["Calibration"][0] == pytest.approx(RESOLUTION)
        assert f["TRIALMAP"]["TrialNo"].tolist() == [7, 8]


def test_header_parsing():
    header = (_chunk(b"MCSHDR", struct.pack("<6H", 10, 0, 0, 4, 0, 0).ljust(64, b"\0"))
              + _chunk(b"STRMHDR", _stream_header("filter", "filt0001", 1000.0, [(0, 15, "16A"), (1, 14, "15A")], 1112))
              + _chunk(b"STRMFMT", _stream_format("V", RESOLUTION, 100)))
    version, streams = parse_streams(header)
    assert version == (10, 0, 0, 4, 0, 0)
    stream = streams[0]
    assert (stream.tag, stream.type_name, stream.sampling_frequency, stream.channel_ids, stream.hwids) == \
        ("filt0001", "filter", 1000.0, ["16A", "15A"], [15, 14])
    assert (stream.unit, stream.ad_zero, stream.resolution, stream.samples_per_block) == ("V", 0x8000, RESOLUTION, 100)
