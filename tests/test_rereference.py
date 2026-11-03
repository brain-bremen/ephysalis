"""probes.<id>.reference_channels: the probe's channels minus the mean of the reference channels."""

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording

from ephysalis.recording_session.probes import apply_probe, rereference
from ephysalis.recording_session.model import ProbeInfo, StreamKind
from ephysalis.recording_session.config import SCHEMA_VERSION, recording_session_config_from_dict


def recording():
    rng = np.random.default_rng(0)
    traces = rng.integers(-1000, 1000, size=(5000, 4)).astype(np.int16)
    rec = NumpyRecording([traces], sampling_frequency=25000.0, channel_ids=["a", "b", "ref1", "ref2"])
    rec.set_channel_gains(0.125)
    rec.set_channel_offsets(0.0)
    return rec, traces.astype(np.float32)


def test_rereference_subtracts_mean_of_references():
    rec, x = recording()
    out = rereference(rec, ["a", "b"], ["ref1", "ref2"])
    assert list(out.channel_ids) == ["a", "b"]
    expected = x[:, :2] - x[:, 2:].mean(axis=1, keepdims=True)
    np.testing.assert_allclose(out.get_traces(start_frame=100, end_frame=900), expected[100:900])
    np.testing.assert_allclose(out.get_traces(channel_ids=["b"])[:, 0], expected[:, 1])
    assert out.get_channel_gains() == pytest.approx([0.125, 0.125])


def test_rereference_rejects_unknown_channels_and_mixed_gains():
    rec, _ = recording()
    with pytest.raises(ValueError, match="not in the stream"):
        rereference(rec, ["a"], ["nope"])
    rec.set_channel_gains([0.125, 0.125, 0.25, 0.125])
    with pytest.raises(ValueError, match="different gains"):
        rereference(rec, ["a"], ["ref1"])


def test_apply_probe_with_reference_channels():
    rec, x = recording()
    info = ProbeInfo(probe_id="v4", streams={StreamKind.AP: "s"}, channel_ids=("a",), reference_channel_ids=("ref1",))
    out = apply_probe(rec, info)
    np.testing.assert_allclose(out.get_traces()[:, 0], x[:, 0] - x[:, 2])


def test_schema_7_reference_channels_and_migration():
    config = recording_session_config_from_dict({"schema_version": 6, "session_id": "s", "subject_id": "m",
                                       "probes": {"v4": {"stream": "elec0003", "channels": ["18C"]}}})
    assert config.schema_version == SCHEMA_VERSION == 9 and config.probes["v4"].reference_channels is None
    config = recording_session_config_from_dict({"schema_version": 7, "session_id": "s", "subject_id": "m",
                                       "probes": {"v4": {"stream": "elec0003", "channels": ["18C"],
                                                         "reference_channels": ["14C"]}}})
    assert config.probes["v4"].reference_channels == ["14C"]
