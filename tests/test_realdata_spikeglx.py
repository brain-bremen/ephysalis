"""Checks against a real session processed by the legacy MATLAB pipeline.

Set ``EPHYSALIS_TEST_SESSION`` to a SpikeGLX session folder that still contains the
``mergedCombined_Uncurated_DataForProbe0.mat`` files written by MATLAB.
"""

import h5py
import numpy as np
import pytest

from ephysalis.io.base import open_session
from ephysalis.stimulus.base import detect_stimulus_system
from conftest import real_session_path

SESSION = real_session_path()
pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(SESSION is None, reason="EPHYSALIS_TEST_SESSION not set"),
]


def _paradigms():
    session = open_session(SESSION)
    for run in session.runs:
        system = detect_stimulus_system(run)
        for paradigm in system.discover(run) if system else []:
            legacy = paradigm.path / "mergedCombined_Uncurated_DataForProbe0.mat"
            if legacy.exists():
                yield pytest.param(system, paradigm, legacy, id=run.name)


@pytest.mark.parametrize("system,paradigm,legacy", list(_paradigms()) if SESSION else [])
def test_conditions_match_matlab(system, paradigm, legacy):
    conditions = system.load_conditions(paradigm)
    with h5py.File(legacy, "r") as f:
        stims = {k: f["stims"][k][()].ravel() for k in f["stims"]}
    n_legacy = len(stims["sampleIndsStart"])
    assert len(conditions) >= n_legacy  # MATLAB truncates to the detected photodiode trials
    for name, values in stims.items():
        if name.startswith("sampleInds"):
            continue
        np.testing.assert_allclose(
            conditions[name].to_numpy(float)[: len(values)], values, err_msg=name
        )
