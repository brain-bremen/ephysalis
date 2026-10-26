"""Calibration, mains removal and CSD (the lab's preprocessing of the epidural recordings):
against literal ports of the MATLAB (``process_channel.m``) and C++ (``dh_csd``) code, and
through the pipeline on a synthetic DAQ session."""

import json

import h5py
import numpy as np
import probeinterface
import pytest
from scipy import signal

from ephyscon.calibration import fit_sine
from ephyscon.cli.main import main
from ephyscon.csd import DH5_FULL_SCALE, c_round, csd_operator
from ephyscon.linenoise import A, B, N_POINTS, line_noise_templates, matlab_int16, reference_quality, zero_crossings
from test_daq import write_v2

FS = 1000.0


def matlab_process_channel(datap: np.ndarray, o_trigs: np.ndarray, npts: int = N_POINTS) -> np.ndarray:
    """Literal port of process_channel.m (0-based; the last cycle runs to the end of the data, a
    cycle never gets more than the 30 template values)."""
    fstate = np.zeros((max(len(A), len(B)) - 1, npts))
    monstate = np.zeros(npts)
    datap = datap.astype(np.int16).copy()
    sam = datap.size
    for i in range(len(o_trigs)):
        if o_trigs[i] + npts <= sam:
            fragm = datap[o_trigs[i]:o_trigs[i] + npts].astype(float)
            for j in range(npts):
                y, fstate[:, j] = signal.lfilter(B, A, [fragm[j]], zi=fstate[:, j])
                monstate[j] = y[0]
            ntp = (o_trigs[i + 1] if i < len(o_trigs) - 1 else sam) - o_trigs[i]
            ntp = min(ntp, npts)
            segment = datap[o_trigs[i]:o_trigs[i] + ntp].astype(float) - monstate[:ntp]
            datap[o_trigs[i]:o_trigs[i] + ntp] = matlab_int16(segment)
    return datap


def mains(n: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """A mains reference with drifting frequency (49.7-50.3 Hz) and its phase."""
    frequency = 50 + 0.3 * np.sin(2 * np.pi * np.arange(n) / FS / 7.0)
    phase = 2 * np.pi * np.cumsum(frequency) / FS
    return np.sin(phase), phase


def test_removal_equals_process_channel_m():
    rng = np.random.default_rng(0)
    ref, phase = mains(20_003)
    x = (rng.standard_normal(ref.size) * 50 + 300 * np.sin(phase) + 150 * np.sin(3 * phase + 0.4)).astype(np.int16)
    crossings = zero_crossings(np.round(ref * 2000).astype(np.int16))
    ours = matlab_int16(x.astype(float) - line_noise_templates(x[:, None].astype(float), crossings)[:, 0])
    np.testing.assert_array_equal(ours, matlab_process_channel(x, crossings))


def test_matlab_int16_rounding():
    np.testing.assert_array_equal(matlab_int16(np.array([0.5, -0.5, 1.5, -2.5, 40000, -40000])),
                                  [1, -1, 2, -3, 32767, -32768])


def test_removal_suppresses_mains_but_not_the_signal():
    rng = np.random.default_rng(1)
    ref, phase = mains(60_000)
    neural = rng.standard_normal(ref.size) * 100
    noise = 500 * (1 + 0.3 * np.sin(2 * np.pi * np.arange(ref.size) / FS / 40)) * (np.sin(phase + 0.7) + 0.4 * np.sin(2 * phase))
    x = np.c_[neural + noise, neural]
    cleaned = x - line_noise_templates(x, zero_crossings(ref))
    late = slice(10_000, ref.size - 100)  # after the ~4 s adaptation
    assert np.std(cleaned[late, 0] - neural[late]) < 0.2 * np.std(noise[late])
    assert np.std(cleaned[late, 1] - neural[late]) < 0.1 * np.std(neural[late])
    fraction, peak = reference_quality(ref)
    assert fraction > 0.95 and peak > 20
    assert reference_quality(rng.standard_normal(ref.size))[0] < 0.5


def cpp_dh_csd_sample(z, xy_in, xy_out, sigm, specres):
    """Literal port of csd_array_generic.cpp / rbf_gaussian.cpp / gauss.c for one sample."""
    n = len(z)
    norm = 1.0 / (0.5 * sigm ** 2)
    a = np.array([[np.exp(-norm * ((xy_in[i] - xy_in[j]) ** 2).sum()) for j in range(n)] for i in range(n)])
    b = np.array(z, float)
    for k in range(n):  # gauss.c: elimination with partial pivoting
        pivot = k + int(np.argmax(np.abs(a[k:, k])))
        a[[k, pivot]], b[[k, pivot]] = a[[pivot, k]], b[[pivot, k]]
        for i in range(k + 1, n):
            m = a[i, k] / a[k, k]
            a[i, k:] -= m * a[k, k:]
            b[i] -= m * b[k]
    lam = np.zeros(n)
    for i in range(n - 1, -1, -1):
        lam[i] = (b[i] - a[i, i + 1:] @ lam[i + 1:]) / a[i, i]
    out = []
    for p in xy_out:  # RbfPrepMulLapl + RbfMulIpol, then -/specres
        d = ((xy_in - p) ** 2).sum(1)
        out.append(-(lam * 4.0 * norm * np.exp(-norm * d) * (norm * d - 1)).sum() / specres)
    return np.array(out)


# Fritz's map (csd_array_fritz.cpp): DAQ channel at each grid position, -1 = no electrode
FRITZ_MAP = [[-1, -1, -1, 4, 1, 21, 13], [-1, -1, -1, 7, 20, 15, 30], [-1, -1, 12, 10, 25, 9, 34],
             [-1, 0, 6, 11, 23, 26, 39], [5, 2, 18, 14, 27, 33, 41], [3, 17, 28, 29, 35, 37, -1],
             [19, 22, 31, 38, 36, -1, -1]]


def fritz_grid(spacing_um: float = 3000.0) -> tuple[list[int], np.ndarray]:
    positions = {c: (j * spacing_um, i * spacing_um) for i, row in enumerate(FRITZ_MAP) for j, c in enumerate(row) if c >= 0}
    channels = sorted(positions)
    return channels, np.array([positions[c] for c in channels])


def test_csd_operator_equals_dh_csd():
    channels, xy_um = fritz_grid()
    inputs = np.array([i for i, c in enumerate(channels) if c != 20])  # map_fritz_oh_20
    xy = xy_um * 1e-6
    operator = csd_operator(xy[inputs], xy, 2.2758 * 0.003, 1.0)
    z = np.random.default_rng(3).standard_normal(len(channels)) * 20e-6
    expected = cpp_dh_csd_sample(z[inputs], xy[inputs], xy, 2.2758 * 0.003, 1.0)
    np.testing.assert_allclose(operator @ z[inputs], expected, rtol=1e-6, atol=1e-9 * np.abs(expected).max())


def test_fit_sine():
    t = np.arange(20_000) / FS
    amplitudes = np.array([300.0, 150.0, 5.0])
    x = np.sin(2 * np.pi * 15.37 * t + 0.3)[:, None] * amplitudes + np.random.default_rng(0).standard_normal((t.size, 3))
    frequency, fitted = fit_sine(x, FS)
    assert frequency == pytest.approx(15.37, abs=0.002)
    np.testing.assert_allclose(fitted, amplitudes, rtol=0.01)


def test_pipeline_calibration_mains_csd(tmp_path):
    """15 Hz calibration run + task run with mains noise on 37 channels of the Fritz grid:
    calibrate, LFP with mains removal, CSD, DH5 export."""
    channels, xy_um = fritz_grid()
    n_chan, mains_channel = 48, 46
    rng = np.random.default_rng(5)
    true_gain_uv = 0.15 + 0.05 * rng.random(n_chan)  # µV at the electrode per count
    t = np.arange(20_000) / FS
    calibration = np.zeros((t.size, n_chan))
    calibration[:, channels] = 100.0 * np.sin(2 * np.pi * 15.4 * t)[:, None] / true_gain_uv[channels]
    calibration[:, 20] = 2.0  # E20 carries no calibration signal
    ref, phase = mains(30_000)
    task = rng.standard_normal((ref.size, n_chan)) * 30
    task[:, channels] += 400 * np.sin(phase + 0.5)[:, None]
    task[:, mains_channel] = 3000 * ref
    session = tmp_path / "ses"
    session.mkdir()
    write_v2(session / "15hz_after.daq", [(0, calibration.astype(np.int16))])
    write_v2(session / "task_a.daq", [(5_000_000_000, task.astype(np.int16))])
    geometry = probeinterface.Probe(ndim=2)
    geometry.set_contacts(positions=xy_um, shapes="circle", shape_params={"radius": 100})
    geometry.set_device_channel_indices(np.arange(len(channels)))
    probeinterface.write_probeinterface(tmp_path / "grid.json", geometry)
    ids = ", ".join(f'"AI{c}"' for c in channels)
    (tmp_path / "session.toml").write_text(
        'schema_version = 4\nsession_id = "s"\nsubject_id = "fritz"\n'
        '[[runs]]\nname = "15hz_after"\ninclude = false\n[[runs]]\nname = "task_a"\n'
        f'[probes.epidural]\nstream = "ADC64 Rev F"\nchannels = [{ids}]\ngeometry = {{ file = "grid.json" }}\n'
        '[probes.epidural.csd]\nexclude_inputs = ["AI20"]\n'
        '[calibration]\nrun = "15hz_after"\n[line_noise]\nenabled = true\n'
        f'[aux_channels.analog]\nmains = {mains_channel}\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["calibrate", str(session), *common]) == 0
    entry = json.loads((tmp_path / "work" / "calibration.json").read_text())["probes"]["epidural"]
    gains = np.array([np.nan if g is None else g for g in entry["gains_uv"]])
    index20 = entry["channels"].index("AI20")
    assert entry["without_signal"] == ["AI20"] and np.isnan(gains[index20])
    keep = np.arange(len(channels)) != index20
    np.testing.assert_allclose(gains[keep], true_gain_uv[channels][keep], rtol=0.01)
    assert main(["lfp", str(session), *common, "--n-jobs", "1"]) == 0
    status = json.loads((tmp_path / "work" / "lfp" / "task_a_epidural" / "line_noise.json").read_text())
    assert status["applied"]
    import spikeinterface.core as si

    lfp = si.load(tmp_path / "work" / "lfp" / "task_a_epidural")
    np.testing.assert_allclose(lfp.get_channel_gains()[keep], gains[keep])  # calibrated
    cleaned = lfp.get_traces(return_in_uV=False).astype(float)[10_000:]
    expected = np.round(task[:, channels]).astype(np.int16)[10_000:].astype(float)
    residual_noise = (cleaned - (expected - 400 * np.sin(phase + 0.5)[10_000:, None])).std()
    assert residual_noise < 0.25 * 400 * np.sqrt(0.5)
    assert main(["csd", str(session), *common, "--n-jobs", "1"]) == 0
    csd = si.load(tmp_path / "work" / "csd" / "task_a_epidural")
    info = json.loads((tmp_path / "work" / "csd" / "task_a_epidural" / "csd.json").read_text())
    assert "AI20" not in info["inputs"] and len(info["outputs"]) == 37
    assert main(["timing", str(session), *common]) == 0
    assert main(["export", str(session), *common, "--output-format", "dh5", "-o", str(tmp_path / "out.dh5")]) == 0
    with h5py.File(tmp_path / "out.dh5") as f:
        block = f["CONT8001"]
        stored = block["DATA"][:, 0]
        values = csd.get_traces(return_in_uV=False)[:, 0].astype(np.float64)
        calibration_value = float(block.attrs["Calibration"][0])
        assert calibration_value == pytest.approx(np.abs(values).max() / DH5_FULL_SCALE)
        np.testing.assert_array_equal(stored, c_round(values / calibration_value).astype(np.int16))
        assert block.attrs["SignalType"] in (b"CSD", "CSD")
    assert main(["export", str(session), *common, "-o", str(tmp_path / "out.nwb")]) == 0


def _common_lfp(n: int, n_chan: int, rng) -> np.ndarray:
    """Spatially common 2-30 Hz signal (300 counts) + independent noise (30 counts)."""
    common = signal.sosfiltfilt(signal.butter(2, (2, 30), btype="band", fs=FS, output="sos"), rng.standard_normal(n))
    return (common / common.std() * 300)[:, None] + rng.standard_normal((n, n_chan)) * 30


def test_channel_quality():
    from ephyscon.quality import channel_quality
    from ephyscon.session_config import ChannelQualityConfig

    channels, xy_um = fritz_grid()
    rng = np.random.default_rng(7)
    x = _common_lfp(60_000, len(channels), rng)
    x[:, 3] = 0  # flat
    x[:, 10] = rng.standard_normal(x.shape[0]) * 3000  # noisy, uncorrelated
    x[:, 12] += 2000 * np.sin(2 * np.pi * 50 * np.arange(x.shape[0]) / FS)  # mains
    x[:20_000:7, 20] = 32500  # clipping
    results = channel_quality(np.round(x).astype(np.int16), xy_um, ChannelQualityConfig(n_windows=50))
    bad = {i: r["status_description"] for i, r in enumerate(results) if r["status"] == "bad"}
    assert sorted(bad) == [3, 10, 12, 20]
    assert "flat" in bad[3] and "power outlier" in bad[10] and "neighbours" in bad[10]
    assert "mains" in bad[12] and "clipping" in bad[20]


def test_quality_restricts_csd(tmp_path):
    """Two runs, a different bad channel in each: the CSD of a run has its good channels only,
    the DH5 CSD block of a channel bad in one run has the other run's region only."""
    channels, xy_um = fritz_grid()
    rng = np.random.default_rng(8)
    session = tmp_path / "ses"
    session.mkdir()
    for k, (name, bad_channel) in enumerate((("task_a", 5), ("task_b", 17))):
        data = np.zeros((40_000, 48))
        data[:, channels] = _common_lfp(40_000, len(channels), rng)
        data[:, bad_channel] = 0
        write_v2(session / f"{name}.daq", [(k * 100_000_000_000, data.astype(np.int16))])
    geometry = probeinterface.Probe(ndim=2)
    geometry.set_contacts(positions=xy_um, shapes="circle", shape_params={"radius": 100})
    geometry.set_device_channel_indices(np.arange(len(channels)))
    probeinterface.write_probeinterface(tmp_path / "grid.json", geometry)
    ids = ", ".join(f'"AI{c}"' for c in channels)
    (tmp_path / "session.toml").write_text(
        'schema_version = 5\nsession_id = "s"\nsubject_id = "fritz"\n'
        f'[probes.epidural]\nstream = "ADC64 Rev F"\nchannels = [{ids}]\ngeometry = {{ file = "grid.json" }}\n'
        '[probes.epidural.csd]\n[probes.epidural.quality]\nn_windows = 40\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    for step in ("lfp", "quality", "csd"):
        assert main([step, str(session), *common, *(["--n-jobs", "1"] if step != "quality" else [])]) == 0
    info = {name: json.loads((tmp_path / "work" / "csd" / f"{name}_epidural" / "csd.json").read_text())
            for name in ("task_a", "task_b")}
    assert "AI5" not in info["task_a"]["outputs"] and "AI5" in info["task_b"]["outputs"]
    assert "AI17" in info["task_a"]["inputs"] and "AI17" not in info["task_b"]["inputs"]
    assert len(info["task_a"]["outputs"]) == len(channels) - 1
    assert main(["timing", str(session), *common]) == 0
    assert main(["export", str(session), *common, "--output-format", "dh5", "-o", str(tmp_path / "out.dh5")]) == 0
    index5, index17 = channels.index(5), channels.index(17)
    with h5py.File(tmp_path / "out.dh5") as f:
        assert len(f[f"CONT{8001 + index5}"]["INDEX"]) == 1 and len(f[f"CONT{8001 + index17}"]["INDEX"]) == 1
        assert len(f[f"CONT{8001 + 0}"]["INDEX"]) == 2
        assert f[f"CONT{8001 + index5}"]["DATA"].shape[0] == 40_000 and f["CONT8001"]["DATA"].shape[0] == 80_000
        lfp5 = f[f"CONT{2001 + index5}"].attrs
        assert lfp5["ChannelStatus"] == "bad" and lfp5["ChannelStatusDescription"].startswith("task_a: flat")
        assert f[f"CONT{2001}"].attrs["ChannelStatus"] == "good"
    bids = tmp_path / "bids"
    assert main(["export", str(session), *common, "--bids", str(bids)]) == 0
    import pandas as pd

    table = pd.read_csv(next(bids.rglob("*_channels.tsv")), sep="\t")
    assert (table.status == "bad").sum() == 2
    assert table.status_description[table.status == "bad"].str.contains("channel selection").all()
