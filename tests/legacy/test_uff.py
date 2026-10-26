"""The DataWave UFF reader (ephyscon.legacy.uff) on synthetic M/P files: samples, block pairing,
event pulses, the slow P clock mapped onto M (steps: :func:`p_clock_fit`), and the DH5 export of
P with its true period."""

import struct

import h5py
import numpy as np
import pytest

from ephyscon.cli.main import main
from ephyscon.io.base import open_session
from ephyscon.legacy.uff import NOMINAL_P_RATE, UffFile, p_clock_fit
from ephyscon.session import StreamKind

PULSE = 660


def write_uff(path, data: np.ndarray, samples_per_record: int = 500) -> None:
    """Header with the UFF magic and an acquisition-script text, then records of 122-byte
    headers + channel-interleaved int16 samples (0.1 ms ticks, ADC scale 4.8828125 mV)."""
    n, n_channels = data.shape
    out = bytearray(b"\x00\x02UFF\x00\x00MS_DOS" + b"\x00" * 50 + b"16 chans cont acq".ljust(401, b"\x00"))  # even length, as in the real files
    for k, start in enumerate(range(0, n - n % samples_per_record, samples_per_record)):
        words = [0] * 61
        words[1] = 0x61
        words[2] = (k * 10 * samples_per_record) % 65536 - (65536 if (k * 10 * samples_per_record) % 65536 > 32767 else 0)
        words[11] = n_channels
        words[12:12 + n_channels] = range(n_channels)
        words[44] = samples_per_record * n_channels
        header = bytearray(struct.pack("<61h", *words))
        struct.pack_into("<f", header, 102, 4.8828125)
        out += header + data[start:start + samples_per_record].astype("<i2").tobytes()
    path.write_bytes(bytes(out))


def _block(rng, n_m=60_000, step_at=30_000, step=3.0):
    """M (1 kHz) and P (NOMINAL_P_RATE slower, +step P samples after M sample step_at) with the
    same pulses on channel 15; LFP-like noise elsewhere."""
    pulses_m = np.cumsum(rng.integers(300, 900, 90)) + 200
    pulses_m = pulses_m[pulses_m < n_m - 10]
    m = rng.integers(-200, 200, (n_m, 16)).astype(np.int16)
    m[:, 15] = 12
    n_p = int(n_m * NOMINAL_P_RATE)
    p = rng.integers(-200, 200, (n_p, 16)).astype(np.int16)
    p[:, 15] = 12
    offset = 4.3
    for t in pulses_m:
        m[t:t + 3, 15] = PULSE
        q = int(round(NOMINAL_P_RATE * t + offset + (step if t >= step_at else 0)))
        if q + 3 < n_p:
            p[q:q + 3, 15] = PULSE
    return m, p


def test_uff_session_and_p_clock(tmp_path):
    rng = np.random.default_rng(0)
    session = tmp_path / "1999-02-12"
    session.mkdir()
    m, p = _block(rng, step=0.0)
    write_uff(session / "MED12021.UFF", m)
    write_uff(session / "PED12021.UFF", p)
    m2, p2 = _block(rng, n_m=20_000, step_at=10**9)
    write_uff(session / "MED12022", m2)  # M file without extension
    write_uff(session / "ED12022.UFF", p2)  # P file without prefix
    file = UffFile(session / "MED12021.UFF")
    assert file.n_channels == 16 and file.adc_scale_mv == pytest.approx(4.8828125) and file.n_samples == 60_000
    np.testing.assert_array_equal(file.traces(1234, 5678), m[1234:5678])

    s = open_session(session)
    assert s.format.name == "uff" and [r.name for r in s.runs] == ["ED12021", "ED12022"]
    run = s.runs[0]
    assert run.streams == ["M", "P"] and run.aux_stream == "M" and run.duration_s == 60.0
    aux = run.get_recording(StreamKind.AUX)
    assert list(aux.channel_ids) == [f"M{c}" for c in range(16)] and aux.get_channel_gains()[0] == pytest.approx(4882.8125)
    events = run.format.read_ttl_events(run)
    assert (events.state == 1).sum() == (np.diff((m[:, 15] > 330).astype(int)) == 1).sum()

    channels = ", ".join(f'"P{c}"' for c in range(14))
    (tmp_path / "session.toml").write_text(
        'schema_version = 5\nsession_id = "19990212"\nsubject_id = "eddi"\n'
        '[probes.arrayM]\nstream = "M"\nchannels = ["M0", "M1"]\n'
        f'[probes.arrayP]\nstream = "P"\nchannels = [{channels}]\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["timing", str(session), *common]) == 0
    assert main(["lfp", str(session), *common, "--n-jobs", "1"]) == 0
    assert main(["export", str(session), *common, "--output-format", "dh5", "--per-run", "-o", str(tmp_path / "o.dh5")]) == 0
    with h5py.File(tmp_path / "o_ED12021.dh5") as f:
        p_block, m_block = f["CONT2003"], f["CONT2001"]  # arrayP's first channel after arrayM's two
        assert m_block.attrs["SamplePeriod"] == 1_000_000 and len(m_block["INDEX"]) == 1
        assert p_block.attrs["SamplePeriod"] == pytest.approx(1e6 / NOMINAL_P_RATE, abs=2)
        index = p_block["INDEX"][()]
        assert len(index) == 1
        # a P pulse lands at the time of its M pulse (within 1 ms)
        pulse_p = int(np.flatnonzero(p[:, 15] == PULSE)[0])
        region = int(np.searchsorted(index["offset"], pulse_p, side="right") - 1)
        t_ns = index["time"][region] + (pulse_p - index["offset"][region]) * p_block.attrs["SamplePeriod"]
        pulse_m = int(np.flatnonzero(m[:, 15] == PULSE)[0])
        assert abs(t_ns / 1e6 - pulse_m) < 1.0


def test_p_clock_steps():
    """One rate, one offset per stretch; a step of the P clock splits the map."""
    pm = np.sort(np.random.default_rng(1).uniform(0, 60_000, 120))
    pp = NOMINAL_P_RATE * pm + 4.3 + np.where(pm > 30_000, 4.0, 0.0)
    p_points, m_points, rate = p_clock_fit(pm, pp)
    assert rate == pytest.approx(NOMINAL_P_RATE, abs=1e-9) and len(p_points) == 4
    np.testing.assert_allclose(p_points, NOMINAL_P_RATE * m_points + 4.3 + np.where(m_points > 30_000, 4.0, 0.0))
