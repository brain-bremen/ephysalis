"""Tests of the sorting helpers (channel selection, concatenation, cutting); Kilosort itself is not run."""

import numpy as np
import pytest
from spikeinterface.core import NumpySorting

from ephyscon.io.base import open_session
from ephyscon.session_config import ProbeConfig
from ephyscon.sorting import RunSegment, concatenate_runs, cut_spikes, load_run_spikes, save_run_spikes, sorting_channels
from ephyscon.session import StreamKind
from ephyscon.sync import ClockMap, RunSync
from conftest import SPIKEGLX_CROP_SESSION

RUN = "3_rfmappingFine_g0"


@pytest.fixture(scope="module")
def session():
    return open_session(SPIKEGLX_CROP_SESSION)


def test_sorting_channels_by_depth_and_bad_channels(session):
    recording = session.run(RUN).get_recording(StreamKind.AP, "imec0")  # channels 82-89, y 820..880 um
    assert sorting_channels(recording, None).size == 8
    selected = sorting_channels(recording, ProbeConfig(insertion_depth_um=850, bad_channels=[1]))
    assert list(selected) == ["imec0.ap#AP82", "imec0.ap#AP84", "imec0.ap#AP85"]


def test_concatenate_runs(session):
    recording, segments = concatenate_runs(session, "imec1", [RUN, RUN], ProbeConfig(insertion_depth_um=760))  # channels 72-79, y 720..780 um
    n = session.run(RUN).get_recording(StreamKind.AP, "imec1").get_num_samples()
    assert segments == [RunSegment(RUN, 0, n), RunSegment(RUN, n, n)]
    assert recording.get_num_samples() == 2 * n
    assert recording.get_num_channels() == 4
    assert recording.has_probe()


def test_cut_spikes_maps_to_reference_clock():
    fs = 30000.0
    segments = [RunSegment("a", 0, 1000), RunSegment("b", 1000, 500)]
    samples = np.array([10, 999, 1000, 1200, 1499])
    sorting = NumpySorting.from_samples_and_labels([samples], [np.array([3, 3, 7, 3, 7])], fs)
    # probe "p1" runs 0.1 % fast and starts 2 ms after the reference in run b
    drift = ClockMap(np.array([0.0, 30030.0]), np.array([0.002, 1.002]), fs)
    syncs = {
        "a": RunSync("p0", {"p0": ClockMap.identity(fs), "p1": ClockMap.identity(fs)}),
        "b": RunSync("p0", {"p0": ClockMap.identity(fs), "p1": drift}),
    }
    runs = cut_spikes(sorting, segments, "p1", syncs)
    assert list(runs["a"]["sample"]) == [10, 999]
    assert list(runs["b"]["sample"]) == [0, 200, 499]
    assert list(runs["b"]["unit_id"]) == [7, 3, 7]
    np.testing.assert_allclose(runs["a"]["time"], [10 / fs, 999 / fs])
    np.testing.assert_allclose(runs["b"]["time"], 0.002 + np.array([0, 200, 499]) / 30030.0)


def test_run_spikes_roundtrip(tmp_path):
    import pandas as pd

    table = pd.DataFrame({"time": [0.1, 0.2], "sample": [3000, 6000], "unit_id": [1, 2]})
    save_run_spikes(tmp_path / "x.npz", table)
    pd.testing.assert_frame_equal(load_run_spikes(tmp_path / "x.npz"), table)
