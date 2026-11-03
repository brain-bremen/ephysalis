"""Calibration of DAQ recordings (ephysalis.formats.input.legacy.daq.calibration): gains from a 15 Hz
calibration-sine run, put on the probe channels of every step (``ephysalis calibrate``, ``[calibration]``)."""

import json

import numpy as np
import pytest

from ephysalis.cli.main import main
from ephysalis.formats.input.legacy.daq.calibration import fit_sine
from test_daq import write_v2

FS = 1000.0  # write_v2: one frame per ms


def test_fit_sine():
    t = np.arange(20_000) / FS
    amplitudes = np.array([300.0, 150.0, 5.0])
    x = np.sin(2 * np.pi * 15.37 * t + 0.3)[:, None] * amplitudes + np.random.default_rng(0).standard_normal((t.size, 3))
    frequency, fitted = fit_sine(x, FS)
    assert frequency == pytest.approx(15.37, abs=0.002)
    np.testing.assert_allclose(fitted, amplitudes, rtol=0.01)


def test_calibrate_and_lfp(tmp_path):
    """Calibration run (100 µV sine at 15.4 Hz, AI3 without signal) and a task run: the gains of
    ``calibration.json`` replace the recorded ones in the LFP."""
    n_chan, rng = 4, np.random.default_rng(1)
    true_gain_uv = np.array([0.15, 0.18, 0.2, 0.17])  # µV at the electrode per count
    t = np.arange(20_000) / FS
    sine = 100.0 * np.sin(2 * np.pi * 15.4 * t)[:, None] / true_gain_uv
    sine[:, 3] = rng.standard_normal(t.size)
    session = tmp_path / "session"
    session.mkdir()
    write_v2(session / "cal.daq", [(0, np.round(sine))])
    write_v2(session / "task.daq", [(0, rng.integers(-300, 300, (3000, n_chan)))], date=(2005, 3, 30, 12, 0, 0))
    ids = ", ".join(f'"AI{c}"' for c in range(n_chan))
    (tmp_path / "session.toml").write_text(
        'schema_version = 4\nsession_id = "s"\nsubject_id = "fritz"\n'
        '[[runs]]\nname = "cal"\ninclude = false\n[[runs]]\nname = "task"\n'
        f'[probes.v1]\nstream = "ADC64 Rev F"\nchannels = [{ids}]\n'
        '[calibration]\nrun = "cal"\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["calibrate", str(session), *common]) == 0
    entry = json.loads((tmp_path / "work" / "calibration.json").read_text())["probes"]["v1"]
    assert entry["without_signal"] == ["AI3"] and entry["gains_uv"][3] is None
    np.testing.assert_allclose(entry["gains_uv"][:3], true_gain_uv[:3], rtol=0.01)
    assert main(["lfp", str(session), *common, "--n-jobs", "1"]) == 0
    import spikeinterface.core as si

    gains = si.load(tmp_path / "work" / "lfp" / "task_v1").get_channel_gains()
    np.testing.assert_allclose(gains[:3], true_gain_uv[:3], rtol=0.01)
    assert np.isnan(gains[3])
