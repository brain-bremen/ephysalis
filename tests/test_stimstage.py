import numpy as np
import pytest

from ephyscon.io.base import open_session
from ephyscon.stimulus.base import detect_stimulus_system
from ephyscon.stimulus.stimstage import get_stim_from_index
from conftest import write_stimstage_paradigm

STIM_FUNC = {"function_handle": {"function": "stimulusStageTest", "type": "simple"}}


def test_get_stim_from_index_is_column_major():
    # MATLAB ind2sub: first field varies fastest
    params = [{"a": np.array([1, 2, 3]), "b": np.array([10, 20]), "c": 5, "stimFunc": STIM_FUNC}]
    expected = [(1, 10), (2, 10), (3, 10), (1, 20), (2, 20), (3, 20)]
    for index, (a, b) in enumerate(expected, start=1):
        param_set, stim = get_stim_from_index(params, index)
        assert param_set == 0
        assert (stim["a"], stim["b"], stim["c"]) == (a, b, 5)
        assert stim["stimFunc"] is STIM_FUNC


def test_get_stim_from_index_spans_parameter_sets():
    params = [
        {"dir": np.array([0.0, 90.0]), "R": np.array([10.0, 20.0])},
        {"dir": np.array([180.0]), "R": 45.0},
    ]
    assert get_stim_from_index(params, 4) == (0, {"dir": 90.0, "R": 20.0})
    assert get_stim_from_index(params, 5) == (1, {"dir": 180.0, "R": 45.0})
    with pytest.raises(IndexError):
        get_stim_from_index(params, 6)


def test_nan_scalar_is_a_single_level():
    _, stim = get_stim_from_index([{"dotColor": np.nan, "x": np.array([1, 2])}], 2)
    assert np.isnan(stim["dotColor"]) and stim["x"] == 2


def test_discover_and_load_conditions(make_spikeglx_session):
    session_path = make_spikeglx_session()
    run_dir = session_path / "3_rfmapping_g0"
    params = [
        {"xc": np.array([-5, 0, 5]), "yc": np.array([1, 2]), "laserAmpl": 0, "stimFunc": "stimulusStageRF"},
        {"xc": 0, "laserAmpl": np.array([1, 3]), "stimFunc": "stimulusStageRF"},
    ]
    order = [1, 6, 7, 8, 2, np.nan, np.nan]  # aborted: trailing NaNs
    write_stimstage_paradigm(run_dir, "1_stimulusStageRF", params, order)

    run = open_session(session_path).runs[0]
    system = detect_stimulus_system(run)
    assert system is not None and system.name == "stimstage"

    (paradigm,) = system.discover(run)
    assert paradigm.name == "RF"
    assert paradigm.metadata["stimulusCount"] == 5
    assert paradigm.metadata["screenInfo"]["FrameRate"] == 60.0

    conditions = system.load_conditions(paradigm)
    assert list(conditions["trial"]) == [0, 1, 2, 3, 4]
    assert list(conditions["condition"]) == [1, 6, 7, 8, 2]
    assert list(conditions["param_set"]) == [0, 0, 1, 1, 0]
    assert list(conditions["xc"]) == [-5, 5, 0, 0, 0]
    assert conditions["yc"].iloc[1] == 2 and np.isnan(conditions["yc"].iloc[2])
    assert list(conditions["laserAmpl"]) == [0, 0, 1, 3, 0]
    assert set(conditions["stimulus_function"]) == {"stimulusStageRF"}


def test_runs_without_stimulus_folder(make_spikeglx_session):
    run = open_session(make_spikeglx_session()).runs[0]
    assert detect_stimulus_system(run) is None
