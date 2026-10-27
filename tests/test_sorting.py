"""Tests of the sorting helpers (channel selection, concatenation, cutting); Kilosort itself is not run."""

import numpy as np
import pytest
from spikeinterface.core import NumpySorting

from ephysalis.io.base import open_session
from ephysalis.session_config import ProbeConfig
from ephysalis.sorting import RunSegment, concatenate_runs, cut_spikes, load_run_spikes, save_run_spikes, sorting_channels
from ephysalis.session import StreamKind
from ephysalis.sync import ClockMap, RunSync
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


def test_sorter_settings_precedence():
    from ephysalis.session_config import SessionConfig, SortingConfig
    from ephysalis.sorting import sorter_settings

    assert sorter_settings(None, "p0") == ("kilosort4", {"clear_cache": True})
    config = SessionConfig(session_id="s", subject_id="m", sorting=SortingConfig(sorter="kilosort4", params={"Th_universal": 8}))
    config.probes["p1"] = ProbeConfig(sorting=SortingConfig(sorter="mountainsort5", params={"detect_threshold": 6}))
    assert sorter_settings(config, "p0") == ("kilosort4", {"clear_cache": True, "Th_universal": 8})
    assert sorter_settings(config, "p1") == ("mountainsort5", {"detect_threshold": 6})
    # command line over config; the configured parameters belong to the configured sorter
    assert sorter_settings(config, "p1", "spykingcircus2", {"a": 1}) == ("spykingcircus2", {"a": 1})
    assert sorter_settings(config, "p0", params={"Th_universal": 9})[1]["Th_universal"] == 9


def test_kilosort_neighbourhoods_limited_to_channel_count():
    from ephysalis.sorting import adapt_sorter_params

    assert adapt_sorter_params("kilosort4", {}, 8) == {"nearest_chans": 8, "nearest_templates": 8}
    assert adapt_sorter_params("kilosort4", {"nearest_chans": 4}, 32) == {"nearest_chans": 4, "nearest_templates": 32}
    assert adapt_sorter_params("kilosort4", {}, 384) == {}
    assert adapt_sorter_params("mountainsort5", {}, 4) == {}
