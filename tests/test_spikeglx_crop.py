"""Tests on the small real SpikeGLX dataset in tests/data/SOMxCD1_172_20250109_crop."""

from fractions import Fraction

import numpy as np
import pytest
from scipy.signal import find_peaks, resample_poly

from ephyscon.auxsignals import AuxSignals, wheel_speed
from ephyscon.io.base import open_session
from ephyscon.session import StreamKind
from ephyscon.session_config import AuxChannelMap, default_session_config
from ephyscon.stimulus.base import detect_stimulus_system
from ephyscon.sync import AUX
from ephyscon.timing import extract_run_timing
from conftest import SPIKEGLX_CROP_SESSION

RUN = "3_rfmappingFine_g0"


@pytest.fixture(scope="module")
def run():
    return open_session(SPIKEGLX_CROP_SESSION).run(RUN)


def _aux_signals(run):
    return AuxSignals(run, AuxChannelMap.legacy_default(run.extra["aux"]["n_analog"]))


def test_session_structure(run):
    assert run.duration_s == pytest.approx(5.0, abs=1e-3)
    assert sorted(run.probes) == ["imec0", "imec1"]
    for probe in run.probes.values():
        assert probe.part_number == "PRB_1_4_0480_1_C"
        assert probe.num_channels == 8
        assert probe.has_lf
    assert run.extra["aux"]["n_analog"] == 8


def test_channel_subset_geometry(run):
    ap = run.get_recording(StreamKind.AP, "imec0")
    assert list(ap.channel_ids) == [f"imec0.ap#AP{c}" for c in range(82, 90)]
    # Neuropixels 1.0 checkerboard: 20 um row pitch, two contacts per row
    np.testing.assert_allclose(ap.get_channel_locations()[:, 1], [820, 820, 840, 840, 860, 860, 880, 880])
    assert ap.get_probe().get_contact_count() == 8


def test_default_aux_roles_match_recording(run):
    """The NIDQ channel roles taken from the MATLAB code fit this rig."""
    roles = AuxChannelMap.legacy_default(run.extra["aux"]["n_analog"])
    nidq = run.get_recording(StreamKind.AUX).get_traces()
    word = nidq[:, -1].astype(np.int32) & 0xFFFF

    def edges(bits: np.ndarray) -> int:
        return int(np.abs(np.diff(bits.astype(np.int8))).sum())

    nidq_sync = (word >> roles.digital["sync"]) & 1
    imec_sync = (run.get_recording(StreamKind.SYNC, "imec0").get_traces()[:, 0] >> 6) & 1
    assert edges(nidq_sync) == edges(imec_sync) == 10  # 1 Hz sync pulse, 5 s

    # photodiode is valid during the whole window and crosses the MATLAB trial threshold
    invalid = (word >> roles.digital["photodiode_invalid"]) & 1
    assert not invalid.any()
    photodiode = nidq[:, roles.analog["photodiode"]]
    assert photodiode.max() > 4000

    assert edges((word >> roles.digital["wheel"]) & 1) > 10


def test_conditions_and_legacy_reference(run):
    system = detect_stimulus_system(run)
    (paradigm,) = system.discover(run)
    assert paradigm.metadata["stimulusCount"] == 6707
    conditions = system.load_conditions(paradigm)
    assert len(conditions) == 6707

    reference = np.load(paradigm.path / "legacy_reference_imec0.npz")
    starts = reference["trial_start_samples"]
    assert len(starts) == 23
    assert starts[0] / 30000 == pytest.approx(0.605, abs=0.01)  # 17.105 s in the full run
    assert reference["running_velocity_1kHz"].shape == (5000,)
    assert reference["lfp_1kHz"].shape == (5000, 8)


@pytest.mark.parametrize("probe_id", ["imec0", "imec1"])
def test_contains_spikes(run, probe_id):
    """The kept channels were chosen for their spiking; legacy Kilosort spikes are visible."""
    import spikeinterface.preprocessing as spre

    ap = run.get_recording(StreamKind.AP, probe_id)
    traces = spre.bandpass_filter(ap, 300, 6000).get_traces().astype(np.float32)
    noise = np.median(np.abs(traces), axis=0) / 0.6745
    crossings = [len(find_peaks(-traces[:, i], height=5 * noise[i], distance=30)[0])
                 for i in range(traces.shape[1])]
    assert min(crossings) > 100  # > 20 Hz multi-unit activity on every channel

    reference = np.load(run.path / "1_stimulusStageRF" / f"legacy_reference_{probe_id}.npz")
    times, peaks = reference["spike_times_s"], reference["spike_peak_channels"]
    assert len(times) > 1000 * 0.9 and len(np.unique(reference["spike_unit_ids"])) >= 20
    channels = list(reference["lfp_channels"])
    samples = np.round(times * ap.sampling_frequency).astype(int)
    visible = [
        -traces[max(i - 30, 0) : i + 30, channels.index(c)].min() > 4 * noise[channels.index(c)]
        for i, c in zip(samples, peaks)
    ]
    assert np.mean(visible) > 0.8


def test_default_config(run):
    session = open_session(SPIKEGLX_CROP_SESSION)
    config = default_session_config(session)
    assert config.subject_id == "SOMxCD1_172"
    assert config.probes["imec0"].serial_number == "23167804621"


@pytest.fixture(scope="module")
def timing(run):
    return extract_run_timing(run)


def test_sync(timing):
    assert timing.sync.reference == "imec0"
    for stream in ("imec1", AUX):
        clock = timing.sync.maps[stream]
        assert clock.source_samples.size == 10
        assert clock.max_residual_s() < 50e-6
    assert timing.sync.maps[AUX].effective_sampling_frequency == pytest.approx(26041.3, abs=0.5)


@pytest.mark.parametrize("probe_id", ["imec0", "imec1"])
def test_trial_times_match_legacy(run, timing, probe_id):
    """Photodiode trial on/offsets agree with the MATLAB pipeline within 5 samples (0.17 ms).

    The remaining difference (about 1 sample for imec0, 4 for imec1) is consistent with the
    MATLAB sync: it smooths the sync offset over 100 edges (~50 s), which near the start of a
    run averages in later clock drift. ephyscon maps every pulse with its neighbouring edges.
    MATLAB reports the first sample above threshold; ephyscon the interpolated crossing.
    """
    (trials,) = timing.trials.values()
    assert len(trials) == 23
    assert trials.attrs["photodiode_mode"] == "square"
    assert list(trials.columns[:4]) == ["trial", "start_time", "stop_time", "condition"]

    reference = np.load(run.path / "1_stimulusStageRF" / f"legacy_reference_{probe_id}.npz")
    for column, legacy in (("start_time", "trial_start_samples"), ("stop_time", "trial_stop_samples")):
        samples = timing.trial_samples("RF", probe_id, column)
        np.testing.assert_allclose(samples, reference[legacy], atol=5)


def test_running_speed(run, timing):
    """Port of get_wheelspeed.m: speed from the encoder pulses (900 ppr, 8 cm radius)."""
    speed = timing.running_speed
    assert speed.shape == (5000,)
    aux = _aux_signals(run)
    n_pulses = aux.wheel_rising_edges().size
    assert speed.sum() / 1000 == pytest.approx(n_pulses * 2 * np.pi * 8 / 900, rel=0.05)


def test_running_speed_legacy_artefact(run):
    """The MATLAB speed is ~8-9x too high: it resampled the encoder line to 30 kHz and
    thresholded it at > 0, so filter ringing adds spurious pulses around every real one.
    Simulating that resampling reproduces the MATLAB output, which validates the port."""
    aux = _aux_signals(run)
    ratio = Fraction(30000 / aux.sampling_frequency).limit_denominator(100)
    resampled = resample_poly(aux.digital("wheel").astype(float), ratio.numerator, ratio.denominator,
                              window=("kaiser", 5.0)) > 0
    rising = np.flatnonzero(~resampled[:-1] & resampled[1:]) + 1
    legacy_like = wheel_speed(rising / 30000, 5.0)

    legacy = np.load(run.path / "1_stimulusStageRF" / "legacy_reference_imec0.npz")["running_velocity_1kHz"]
    interior = slice(500, 4500)  # away from the smoothing edges of the crop
    assert np.corrcoef(legacy_like[interior], legacy[interior])[0, 1] > 0.95
    assert rising.size > 5 * aux.wheel_rising_edges().size


def test_sync_edge_cache(run, tmp_path, monkeypatch):
    import ephyscon.sync as sync_module
    from ephyscon.sync import SyncEdgeCache, load_or_extract_edges

    channels = AuxChannelMap.legacy_default(8)
    cache = SyncEdgeCache(tmp_path)
    extracted = load_or_extract_edges(run, channels, cache)
    assert cache.path(RUN).exists()

    def fail(*args, **kwargs):
        raise AssertionError("edges should come from the cache")

    monkeypatch.setattr(sync_module, "digital_edges", fail)
    cached = load_or_extract_edges(run, channels, cache)
    for stream in extracted:
        np.testing.assert_array_equal(cached[stream].samples, extracted[stream].samples)
        np.testing.assert_array_equal(cached[stream].rising, extracted[stream].rising)
    timing = extract_run_timing(run, sync_cache=cache)
    assert len(timing.trials["RF"]) == 23

    # another NIDQ sync bit invalidates the aux entry only
    other = channels.model_copy(deep=True)
    other.digital["sync"] = 5
    calls = []
    monkeypatch.setattr(sync_module, "digital_edges", lambda rec, ch, bit, **kw: calls.append(bit) or extracted[AUX])
    load_or_extract_edges(run, other, cache)
    assert calls == [5]


def test_cli_sync(tmp_path, capsys):
    from ephyscon.cli.main import main

    assert main(["sync", str(SPIKEGLX_CROP_SESSION), "--cache-dir", str(tmp_path)]) == 0
    assert "imec0 10, imec1 10, aux 10" in capsys.readouterr().out
    assert (tmp_path / f"{RUN}.npz").exists()


def test_cli_timing_and_lfp(tmp_path, capsys):
    from ephyscon.cli.main import main

    assert main(["timing", str(SPIKEGLX_CROP_SESSION), "--work-dir", str(tmp_path)]) == 0
    assert "RF: 23 trials" in capsys.readouterr().out
    assert (tmp_path / "trials" / f"{RUN}_RF.csv").exists()
    assert np.load(tmp_path / "running_speed" / f"{RUN}.npy").shape == (5000,)
    assert main(["lfp", str(SPIKEGLX_CROP_SESSION), "--work-dir", str(tmp_path), "--n-jobs", "1"]) == 0
    assert "imec1: LFP 8 channels, 5 s" in capsys.readouterr().out
