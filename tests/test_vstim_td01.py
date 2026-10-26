"""Trials of DAQ runs from TD01 + .tdr (ephyscon.stimulus.vstim_td01) on a synthetic DAQ file
and a synthetic VStim 1.3 .tdr. On real data (Fritz 2005-03-30 runs a and e, 2006-09-01 run c)
the trials equal the TRIALMAP of the BIDS conversions (numbers, stimuli, outcomes, times)."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from ephyscon.cli.main import main
from ephyscon.io.base import open_session
from ephyscon.stimulus.base import detect_stimulus_system
from ephyscon.stimulus.vstim_td01 import TRIAL_END_AFTER_OUTCOME_PULSE_MS, tdr_date
from test_daq import write_v2

T0 = 13_127_830_000_000  # DAQ clock (ns) at the first sample
MS = 1_000_000


def write_tdr(path: Path, trials: list[tuple[int, int, int, float]], day: str = "30.03.2005") -> Path:
    """VStim 1.3 .tdr ($FH1 v2, $TH1/$TS1 v1): ``trials`` = (number, stimulus, outcome code,
    end in s = positive transition of virtual trigger 11)."""
    lines = ["$FH1   4   2", "1.3\t\t// VStim program version", "1.01\t\t// .tdr file format version",
             f"{day}\t11:49:37\t100\t// date, start time, refresh rate"]
    for number, stimulus, outcome, end in trials:
        pairs = []
        for k in range(32):
            positive = {0: 0.0, 11: end}.get(k, -0.01)
            negative = {0: 0.5, 10: end}.get(k, -0.01)
            pairs += [f"{positive:.4f}", f"{negative:.4f}"]
        lines += [f"$TH1   2   1    {number}    {stimulus}    1    {outcome}    0     -1.00",
                  f"$TS1   1   1    {number}  11:49:40    " + "  ".join(pairs)]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


def _session(folder: Path, events, trials) -> Path:
    folder.mkdir()
    data = np.zeros((20_000, 4), dtype=np.int16)
    write_v2(folder / "fritz_2005_03_30_task_a.daq", [(T0, data)], events=events, trials=trials)
    return folder


def test_trials_from_td01_and_tdr(tmp_path):
    starts = [1000, 6000, 12000]  # ms after the first sample
    events = []
    for s in starts:
        events += [(T0 + s * MS, 1, 1), (T0 + (s + 300) * MS, -1, 0)]
    events += [(T0 + 14000 * MS, 14, 0), (T0 + 14500 * MS, -14, 0)]  # reward pulse of the trial without .tdr record
    td01 = [(T0 + (s + 2) * MS, number, stimulus) for s, number, stimulus in zip(starts, (11, 12, 13), (6, 5, 4))]
    session = _session(tmp_path / "ses", sorted(events), td01)
    write_tdr(session / "2005_03_30.tdr", [(11, 6, 7, 1.2), (12, 5, 1, 2.5)])
    assert tdr_date(session / "2005_03_30.tdr").isoformat() == "2005-03-30"
    (tmp_path / "session.toml").write_text(
        'schema_version = 4\nsession_id = "s"\nsubject_id = "fritz"\n'
        '[aux_channels.digital]\ntrial_start = 0\neye_error = 10\nreward = 13\n')
    run = open_session(session).runs[0]
    assert detect_stimulus_system(run).name == "vstim_td01"
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["timing", str(session), *common]) == 0
    trials = pd.read_csv(next((tmp_path / "work" / "trials").glob("*.csv")))
    assert trials.vstim_trial.tolist() == [11, 12, 13] and trials.condition.tolist() == [6, 5, 4]
    assert trials.outcome.tolist() == [7, 1, 1]  # EyeErr, Hit (.tdr); Hit (reward pulse)
    assert trials.outcome_source.tolist() == ["tdr", "tdr", "trigger lines"]
    np.testing.assert_allclose(trials.start_time, np.array(starts) / 1000)
    np.testing.assert_allclose(trials.stop_time, [1.0 + 1.2, 6.0 + 2.5, 14.0 + TRIAL_END_AFTER_OUTCOME_PULSE_MS / 1000])
    assert main(["export", str(session), *common, "--output-format", "dh5", "-o", str(tmp_path / "out.dh5")]) == 0
    with h5py.File(tmp_path / "out.dh5") as f:
        assert f["TRIALMAP"]["TrialNo"].tolist() == [11, 12, 13] and f["TRIALMAP"]["Outcome"].tolist() == [7, 1, 1]


def test_trial_end_line_and_no_tdr(tmp_path):
    events = [(T0 + 1000 * MS, 1, 1), (T0 + 1300 * MS, -1, 0), (T0 + 3000 * MS, 12, 0), (T0 + 3166 * MS, -12, 0),
              (T0 + 5000 * MS, 1, 1), (T0 + 5300 * MS, -1, 0)]
    td01 = [(T0 + 1002 * MS, 40, 2), (T0 + 5002 * MS, 41, 3), (T0 + 500 * MS, 39, 1)]  # 39: before any start edge
    session = _session(tmp_path / "ses", sorted(events), sorted(td01))
    (tmp_path / "session.toml").write_text(
        'schema_version = 4\nsession_id = "s"\nsubject_id = "fritz"\n'
        '[aux_channels.digital]\ntrial_start = 0\ntrial_end = 11\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["timing", str(session), *common]) == 0
    trials = pd.read_csv(next((tmp_path / "work" / "trials").glob("*.csv")))
    assert trials.vstim_trial.tolist() == [40, 41]
    assert trials.outcome.tolist() == [-1, -1] and trials.stop_source.tolist() == ["trial_end line", "start"]
    np.testing.assert_allclose(trials.stop_time, [3.0, 5.0])
