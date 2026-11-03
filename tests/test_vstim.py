"""VStim plugin on the Open Ephys test session (NI-DAQ aux stream, VStim messages, TDR file)."""

import numpy as np
import pandas as pd
import pytest

from conftest import DATA_FOLDER, skip_if_no_data
from conftest import extract_all
from ephysalis.cli.main import main
from ephysalis.formats.output.dh5 import EYE_CONT_IDS
from ephysalis.formats.input.base import open_session
from ephysalis.metadata.base import run_trial_tables
from ephysalis.timeline.steps import extract_run_timing


pytestmark = skip_if_no_data
RUN = "experiment1_recording1"


@pytest.fixture(scope="module")
def timing():
    return extract_run_timing(open_session(DATA_FOLDER).run(RUN))


def test_vstim_is_detected():
    plugin, tables = run_trial_tables(open_session(DATA_FOLDER).run(RUN), None, None)
    assert plugin is not None and plugin.name == "vstim" and list(tables) == ["Test"]


def test_trials_from_tdr_and_trigger_line(timing):
    assert list(timing.trials) == ["Test"]
    trials = timing.trials["Test"]
    assert trials.attrs["vstim_version"] == "2.49"
    assert len(trials) == 13 and set(trials.start_source) == {"trigger"}
    assert list(trials.condition[:4]) == [3, 0, 1, 0]
    assert trials.outcome_name[2] == "Late" and trials.outcome_name[12] == "Cancelled"
    # trial start line 1 rises at 2.7678 s (NI TTL), the TRIAL_START message follows at 2.9244 s
    assert trials.start_time[0] == pytest.approx(2.7678, abs=1e-3)
    np.testing.assert_allclose(trials.stop_time - trials.start_time, trials.duration_ms / 1000)
    assert (trials.start_time.diff().dropna() > 0).all()


def test_eye_position(timing):
    eye = timing.eye_position
    assert eye.shape[1] == 2 and abs(eye.shape[0] - 59140) <= 1  # 1 kHz over the 59.14 s run
    assert np.isfinite(eye).all() and eye.max() < 10 and eye.std(axis=0).min() > 0.1  # volts


@pytest.mark.parametrize("output_format", ["nwb", "dh5"])
def test_timing_and_export(tmp_path, output_format):
    pytest.importorskip("dh5io")
    work = tmp_path / "work"
    assert extract_all([str(DATA_FOLDER), "--work-dir", str(work)]) == 0
    assert len(pd.read_csv(work / "trials" / f"{RUN}_Test.csv")) == 13
    assert (work / "eye" / f"{RUN}.npy").exists()
    if output_format == "dh5":
        import dh5io

        if not hasattr(dh5io, "Spike"):
            pytest.skip("needs dh5io with SPIKE block writing")
    out = tmp_path / f"session.{output_format}"
    assert main(["export", str(DATA_FOLDER), "--work-dir", str(work), "--output-format", output_format, "-o", str(out)]) == 0
    if output_format == "nwb":
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(str(out), "r") as io:
            nwbfile = io.read()
            assert len(nwbfile.trials) == 13 and set(nwbfile.trials["outcome_name"][:]) >= {"Hit", "Cancelled"}
            series = nwbfile.processing["behavior"]["EyeTracking"][f"eye_position_{RUN}"]
            assert series.data.shape[1] == 2
    else:
        import dh5io

        with dh5io.DH5File(out, "r") as file:
            trialmap = file.get_trialmap()
            assert len(trialmap) == 13 and list(trialmap["Outcome"][[2, 12]]) == [6, 10]
            assert file.get_cont_group_by_id(EYE_CONT_IDS[0]).name == "eye position x"
