"""Tests on the small real SpikeGLX dataset in tests/data/SOMxCD1_172_20250109_crop."""

import numpy as np
import pytest
from scipy.signal import find_peaks

from ephyscon.io.base import open_session
from ephyscon.session import StreamKind
from ephyscon.session_config import AuxChannelMap, default_session_config
from ephyscon.stimulus.base import detect_stimulus_system
from conftest import SPIKEGLX_CROP_SESSION

RUN = "3_rfmappingFine_g0"


@pytest.fixture(scope="module")
def run():
    return open_session(SPIKEGLX_CROP_SESSION).run(RUN)


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
