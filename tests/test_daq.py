"""The DAQ reader (ephyscon.io.daq) on synthetic files of file versions 1.0, 2.0 and 3.3.

The real recordings (Fritz 2004-2006, Walter 2000) may not be published; ``ephyscon.io.daq`` was
checked against the readers of the BIDS conversions on them (samples, events and trial
descriptors identical, board fit as in ``clocksync.card_fit``)."""

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ephyscon.cli.main import main
from ephyscon.io.base import open_session
from ephyscon.io.daq import DaqFormat, board_fit, daq_file, read_index
from ephyscon.session import StreamKind
from ephyscon.sync import AUX


def _record(tag: bytes, body: bytes) -> bytes:
    return struct.pack("<4si", tag, 8 + len(body)) + body


def _header(version: tuple[int, int], date=(2005, 3, 30, 11, 49, 37)) -> bytes:
    return (_record(b"VER1", struct.pack("<4h", 1, 0, *version)) + _record(b"DATE", struct.pack("<6h", *date))
            + _record(b"NAME", b"User".ljust(40, b"\0") + b"Subject".ljust(40, b"\0")))


def _boards(*names: str) -> bytes:
    return _record(b"BORD", struct.pack("<i", len(names)) + b"".join(n.encode().ljust(20, b"\0") for n in names))


def write_v2(path: Path, frames: list[tuple[int, np.ndarray]], events=(), trials=(), version=(2, 0),
             truncate_fcr1_size=False, zero_tail=0) -> Path:
    """Version 1.x/2.0 file: ``frames`` = [(time ns, samples (n, channels))] blocks of CR01 records
    (1 ms apart); ``events`` (time, ±line, pattern) and ``trials`` (time, trial, stim) are
    written after the frame block whose start is the last before their time."""
    n_chan = frames[0][1].shape[1]
    fcr1 = struct.pack("<ffhh", 10.0, -10.0, n_chan, 16) + b"".join(struct.pack("<hi", c, 0) for c in range(n_chan))
    fcr1_record = bytearray(_record(b"FCR1", fcr1))
    if truncate_fcr1_size:  # as in Walter's 2000 files
        struct.pack_into("<i", fcr1_record, 4, len(fcr1_record) & 0xFF)
    out = bytearray(_header(version) + _boards("ADC64 Rev F") + fcr1_record)
    pending = sorted([(t, b"EV01", struct.pack("<qiI", t, line, pattern)) for t, line, pattern in events]
                     + [(t, b"TD01", struct.pack("<qiiII", t, trial, stim, 0, 0)) for t, trial, stim in trials])
    for t0, samples in frames:
        for k, frame in enumerate(samples):
            t = t0 + k * 1_000_000
            while pending and pending[0][0] <= t:
                _, tag, body = pending.pop(0)
                out += _record(tag, body)
            out += _record(b"CR01", struct.pack("<q", t) + frame.astype("<i2").tobytes())
    for _, tag, body in pending:
        out += _record(tag, body)
    path.write_bytes(bytes(out) + b"\0" * zero_tail)
    return path


def test_v2_runs_samples_and_events(tmp_path):
    rng = np.random.default_rng(0)
    a, b = rng.integers(-3000, 3000, (500, 8)), rng.integers(-3000, 3000, (300, 8))
    t0 = 13_127_830_000_000  # the DAQ clock need not start at 0
    t_b = t0 + 3_000_000_000  # paused for 2.5 s
    events = [(t0 + 100_000_000, 1, 1), (t0 + 150_000_000, -1, 0), (t0 + 200_000_000, 4, 8), (t_b + 20_000_000, -4, 0)]
    write_v2(tmp_path / "rec_a.daq", [(t0, a), (t_b, b)], events=events, trials=[(t0 + 100_000_000, 7, 3)])
    session = open_session(tmp_path)
    assert isinstance(session.format, DaqFormat)
    assert [r.name for r in session.runs] == ["rec_a_part1", "rec_a_part2"]
    first, second = session.runs
    assert (second.start_time - first.start_time).total_seconds() == pytest.approx(3.0)
    assert first.duration_s == pytest.approx(0.5) and first.aux_stream == "ADC64 Rev F"
    recording = first.get_recording(StreamKind.AUX)
    assert recording.sampling_frequency == 1000 and list(recording.channel_ids[:2]) == ["AI0", "AI1"]
    np.testing.assert_array_equal(recording.get_traces(), a)
    np.testing.assert_array_equal(second.get_recording(StreamKind.AUX).get_traces(start_frame=10, end_frame=20,
                                                                                   channel_ids=["AI3"])[:, 0], b[10:20, 3])
    np.testing.assert_allclose(recording.get_channel_gains(), 20 / 65536 * 1e6)
    ttl = first.format.read_ttl_events(first)
    assert ttl.to_dict("list") == {"aux_sample": [100, 150, 200], "line": [1, 1, 4], "state": [1, 0, 1]}
    assert second.format.read_ttl_events(second).to_dict("list") == {"aux_sample": [20], "line": [4], "state": [0]}
    np.testing.assert_array_equal(np.flatnonzero(np.diff(first.format.read_aux_digital(first, 0).astype(int))) + 1,
                                  [100, 150])
    assert first.format.read_trial_descriptors(first).to_dict("list") == {
        "aux_sample": [100], "trial_number": [7], "stimulus_number": [3]}
    assert first.format.aux_analog_gain_v(first, 2) == pytest.approx(20 / 65536)


def test_v1_truncated_fcr1_size_and_zero_tail(tmp_path):
    """Version 1.0 (2000): FCR1 size field holds only its low byte; preallocated files end in zeros."""
    data = np.arange(64 * 40, dtype=np.int16).reshape(40, 64)
    path = write_v2(tmp_path / "walt.daq", [(1_204_000_000, data)], version=(1, 0), truncate_fcr1_size=True,
                    zero_tail=4096)
    index = read_index(str(path))
    assert index.header.file_version == (1, 0) and len(index.header.channels) == 64
    assert index.zero_tail == path.stat().st_size - 4096
    run = open_session(tmp_path).runs[0]
    np.testing.assert_array_equal(run.get_recording(StreamKind.AUX).get_traces(), data)


def write_v3(path: Path, rate_hz: float = 1000.0, record: int = 100, n_records: int = 30, fit=(-50e6, 1.0038),
             garbage_after: int | None = None, cut_last: int = 0, gap_after: int | None = None) -> Path:
    """Version 3.3 file of the 2006 layout: board 1 (ADC64, channels 0-63) records nTrodes 0
    (channels 0-1) and 5; board 2 (MC-128, 64-191) nTrode 100, which runs ``fit[1]`` times
    faster; trigger line 1 on board 1 is copied to line 17 on board 2 (``fit``: board 2 time
    = offset + rate x board 1 time). Channel c carries ``c * 100 + sample index`` (board 1)."""
    period = int(round(1e9 / rate_hz))
    t0 = 1_199_960_000
    ntrodes = [(0, 0, 0, 2), (5, 0, 5, 1), (100, 0, 100, 1)]
    edr1 = struct.pack("<i", len(ntrodes)) + b"".join(struct.pack("<hbhbhhh", i, k, f, n, 32, 8, 16) for i, k, f, n in ntrodes)
    cdr1 = struct.pack("<i", 192) + b"".join(
        struct.pack("<hhh4f", c, c if c < 64 else c - 64, 16 if c < 64 else 12, 5.0 if c < 64 else 4.0,
                    -5.0 if c < 64 else -4.0, 1.0, rate_hz) for c in range(192))
    out = bytearray(_header((3, 3), (2006, 9, 1, 12, 49, 46)) + _record(b"RECP", struct.pack("<Hbbi", 0, 11, 0, 0))
                    + _boards("ADC64 F/H - Board 1", "MC-128 - Board 1") + _record(b"EDR1", edr1) + _record(b"CDR1", cdr1))
    # trial-start pulses at irregular intervals (periodic pulses would make the board fit ambiguous)
    gaps = np.random.default_rng(1).integers(150_000_000, 400_000_000, 1000)
    end = t0 + n_records * record * period - 500_000_000
    pulses = [int(t) for t in t0 + 30_000_000 + np.cumsum(gaps) if t < end]
    pause = 0
    for r in range(n_records):
        if gap_after is not None and r == gap_after:
            pause = 5_000_000_000
        t = t0 + r * record * period + pause
        first = r * record
        for nid, _, first_chan, n_chan in ntrodes:
            samples = np.stack([(first_chan + c) * 100 + np.arange(first, first + record) for c in range(n_chan)], axis=1)
            out += _record(b"CR02", struct.pack("<qH", t, nid) + (samples * (16 if nid == 100 else 1)).astype("<i2").tobytes())
        for p in pulses:
            if t <= p < t + record * period:
                out += _record(b"EV02", struct.pack("<qi", p, 1)) + _record(b"EV02", struct.pack("<qi", p + 10_000_000, -1))
                out += _record(b"EV02", struct.pack("<qi", int(fit[0] + fit[1] * p), 17))
                out += _record(b"EV02", struct.pack("<qi", int(fit[0] + fit[1] * (p + 10_000_000)), -17))
        if r == 3:
            out += _record(b"TD01", struct.pack("<qiiII", t + 1_000_000, 12, 4, 0, 0))
        if garbage_after is not None and r == garbage_after:
            out += b"\xab" * 1000
    path.write_bytes(bytes(out[:len(out) - cut_last]))
    return path


def test_v3_boards_and_clock_maps(tmp_path):
    write_v3(tmp_path / "f_task_a.daq")
    run = open_session(tmp_path).runs[0]
    assert run.extra["streams"] == ["ADC64 F/H - Board 1", "MC-128 - Board 1"]
    board1 = run.get_recording(StreamKind.AUX)
    assert list(board1.channel_ids) == ["AI0", "AI1", "AI5"] and board1.get_num_samples() == 3000
    np.testing.assert_array_equal(board1.get_traces(start_frame=95, end_frame=105, channel_ids=["AI1"])[:, 0],
                                  100 + np.arange(95, 105))
    np.testing.assert_allclose(board1.get_channel_gains(), 10 / 65536 * 1e6)
    fit = daq_file(run.extra["file"]).fit
    assert fit["rate"] == pytest.approx(1.0038, abs=1e-6) and fit["offset_ns"] == pytest.approx(-50e6, abs=2e3)
    # the MC-128 channel as a probe; its samples are left-aligned 12-bit counts
    run.probes["v1"] = __import__("ephyscon.session", fromlist=["ProbeInfo"]).ProbeInfo(
        probe_id="v1", streams={StreamKind.AP: "MC-128 - Board 1"})
    v1 = run.get_recording(StreamKind.AP, "v1")
    np.testing.assert_allclose(v1.get_channel_gains(), 8 / 65536 * 1e6)
    sync = run.format.clock_maps(run, AUX)
    k = np.array([0, 1234, v1.get_num_samples() - 1])
    board2_ns = 1_199_960_000 + k * 1_000_000  # nominal board-2 times of these samples
    start_ns = 1_199_960_000  # the run starts with board 1
    expected_s = ((board2_ns - fit["offset_ns"]) / fit["rate"] - start_ns) / 1e9
    np.testing.assert_allclose(sync.to_reference_time("v1", k), expected_s, atol=1e-9)
    np.testing.assert_allclose(sync.to_reference_time(AUX, [0, 1000]), [0.0, 1.0])
    # copies of line 1 on board 2 land on the board-1 edges
    ttl = run.format.read_ttl_events(run)
    line1, line17 = ttl[ttl.line == 1], ttl[ttl.line == 17]
    assert len(line1) == len(line17) > 10
    np.testing.assert_allclose(line17.aux_sample.to_numpy(), line1.aux_sample.to_numpy(), atol=1)
    assert run.format.read_trial_descriptors(run).trial_number.tolist() == [12]


def test_v3_damaged_bytes_cut_end_and_pause(tmp_path):
    path = write_v3(tmp_path / "f_damaged.daq", garbage_after=10, cut_last=150)
    index = read_index(str(path))
    assert len(index.damaged) == 1 and index.damaged[0][1] == 1000
    assert index.truncated_bytes == 150
    run = open_session(tmp_path).runs[0]
    traces = run.get_recording(StreamKind.AUX).get_traces(channel_ids=["AI0"])[:, 0]
    np.testing.assert_array_equal(traces, np.arange(traces.size))  # nothing lost before the damage or the cut
    (tmp_path / "pause").mkdir()
    write_v3(tmp_path / "pause" / "f_pause.daq", gap_after=12)
    runs = open_session(tmp_path / "pause").runs
    assert [r.name for r in runs] == ["f_pause_part1", "f_pause_part2"]
    assert [r.get_recording(StreamKind.AUX).get_num_samples() for r in runs] == [1200, 1800]
    np.testing.assert_array_equal(runs[1].get_recording(StreamKind.AUX).get_traces(end_frame=3, channel_ids=["AI5"])[:, 0],
                                  500 + np.arange(1200, 1203))


def test_index_cache(tmp_path):
    """The record index is kept on disk and read back identically by a new process."""
    from ephyscon.io import daq

    path = str(write_v3(tmp_path / "f.daq", garbage_after=4, cut_last=10))
    scanned = read_index(path)
    cached = daq._index_cache_path(path, daq._version(path))
    assert cached.exists()
    daq._read_index.cache_clear()  # as in a new process
    loaded = read_index(path)
    assert loaded is not scanned
    assert (loaded.damaged, loaded.truncated_bytes, loaded.zero_tail) == (scanned.damaged, scanned.truncated_bytes, None)
    np.testing.assert_array_equal(loaded.events, scanned.events)
    np.testing.assert_array_equal(loaded.trials, scanned.trials)
    for nid, pieces in scanned.pieces.items():
        for name in ("data_offsets", "n_samples", "strides", "times", "starts"):
            np.testing.assert_array_equal(getattr(loaded.pieces[nid], name), getattr(pieces, name))
        assert (loaded.pieces[nid].channels, loaded.pieces[nid].period_ns) == (pieces.channels, pieces.period_ns)


def test_board_fit_without_copies():
    assert board_fit(np.array([[0, 1], [10, -1]])) is None


def test_recording_is_serializable(tmp_path):
    from spikeinterface.core import load

    write_v3(tmp_path / "f.daq")
    recording = open_session(tmp_path).runs[0].get_recording(StreamKind.AUX)
    np.testing.assert_array_equal(load(recording.to_dict()).get_traces(end_frame=20), recording.get_traces(end_frame=20))


def test_pipeline(tmp_path):
    """Probe on two board-1 channels at 25 kHz, through LFP, MUA, DH5 (with raw data) and NWB."""
    session = tmp_path / "session"
    session.mkdir()
    write_v3(session / "f_task_a.daq", rate_hz=25000.0, record=25000, n_records=3)
    (tmp_path / "session.toml").write_text(
        'schema_version = 3\nsession_id = "s"\nsubject_id = "fritz"\n'
        '[probes.v4]\nstream = "ADC64 F/H - Board 1"\nchannels = ["AI0", "AI1"]\n'
        '[aux_channels.digital]\ntrial_start = 0\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["timing", str(session), *common]) == 0
    assert main(["lfp", str(session), *common, "--n-jobs", "1"]) == 0
    assert main(["mua", str(session), *common, "--n-jobs", "1"]) == 0
    assert main(["export", str(session), *common, "--output-format", "dh5", "--raw", "-o", str(tmp_path / "out.dh5")]) == 0
    assert main(["export", str(session), *common, "-o", str(tmp_path / "out.nwb")]) == 0
    events = pd.read_csv(tmp_path / "work" / "events" / "f_task_a.csv")
    assert set(events.line) == {1, 17}
    import h5py

    with h5py.File(tmp_path / "out.dh5") as f:
        np.testing.assert_array_equal(f["CONT2"]["DATA"][:100, 0], 100 + np.arange(100))  # raw AI1
        assert f["CONT1601"].attrs["Name"] == b"aux AI5"  # the board-1 channel outside the probe
        assert "CONT2001" in f and "CONT4001" in f
