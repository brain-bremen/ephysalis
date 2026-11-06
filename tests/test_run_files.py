"""The files of a run in a work dir, when another run's name starts with "<run>_" (Botox 2010-11-10: runs
20101110_task and 20101110_task_aftercrash; the export of the first took the trials of both)."""

from pathlib import Path

import pandas as pd

from ephysalis.filesystem import run_files
from ephysalis.formats.output.nwb import recording_session_trials, single_run_recording_session
from ephysalis.recording_session.model import RecordingSession, Run


def _session(tmp_path: Path) -> RecordingSession:
    runs = [Run(name=name, path=tmp_path, index=k, format=None, duration_s=100.0)
            for k, name in enumerate(["day_task", "day_task_aftercrash", "day_dotmap"])]
    return RecordingSession(name="day", path=tmp_path, format=None, runs=runs)


def _trials(folder: Path, name: str, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"TrialNo": range(n), "start_time": [float(k) for k in range(n)],
                  "stop_time": [k + 0.5 for k in range(n)]}).to_csv(folder / name, index=False)


def test_run_files_leave_out_the_runs_with_longer_names(tmp_path):
    for name in ("day_task_botox_task.csv", "day_task_aftercrash_botox_task.csv", "day_dotmap_botox_task.csv"):
        _trials(tmp_path, name, 1)
    names = ("day_task", "day_task_aftercrash", "day_dotmap")
    assert [p.name for p in run_files(tmp_path, "day_task", ".csv", names)] == ["day_task_botox_task.csv"]
    assert [p.name for p in run_files(tmp_path, "day_task_aftercrash", ".csv", names)] == [
        "day_task_aftercrash_botox_task.csv"]
    assert len(run_files(tmp_path, "day_task", ".csv")) == 2  # without the run names: both (the old behaviour)


def test_the_trials_of_a_run_and_of_a_file_per_run(tmp_path):
    session = _session(tmp_path)
    _trials(tmp_path / "trials", "day_task_botox_task.csv", 3)
    _trials(tmp_path / "trials", "day_task_aftercrash_botox_task.csv", 2)
    assert session.runs[0].recording_run_names == ("day_task", "day_task_aftercrash", "day_dotmap")
    offsets = {run.name: 0.0 for run in session.runs}
    one = single_run_recording_session(session, session.runs[0])  # export --per-run: the session of one run
    trials = recording_session_trials(tmp_path, one.runs, offsets)
    assert len(trials) == 3 and set(trials.experimental_protocol) == {"botox_task"}
    assert one.runs[0].recording_run_names == session.runs[0].recording_run_names
    all_runs = recording_session_trials(tmp_path, session.runs, offsets)
    assert list(all_runs.groupby("run").size().items()) == [("day_task", 3), ("day_task_aftercrash", 2)]
