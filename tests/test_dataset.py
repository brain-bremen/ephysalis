"""dataset.toml (defaults, per-session override files and inline overrides) and ``ephysalis run``: steps
done once, reruns after config changes of only the affected session, pause and resume, locks, skipped
steps, code changes."""

import json
from pathlib import Path

import pytest
from conftest import SPIKEGLX_CROP_RECORDING_SESSION

from ephysalis.cli.main import main
from ephysalis.dataset.config import load_dataset, merge_tables, step_argv
from ephysalis.dataset.runner import (
    PAUSE_FILE,
    Runner,
    read_record,
    request_pause,
    status_table,
)

STEPS = ["align-clocks", "extract-events", "lfp"]


def write_dataset(root: Path, overrides_b: str = "", extra: str = "", steps=STEPS) -> Path:
    (root / "recording_sessions").mkdir(exist_ok=True)
    (root / "recording_sessions" / "b.toml").write_text('schema_version = 8\nnotes = "from the file"\n'
                                              '[probes.imec0]\narea = "V2"\n')
    path = root / "dataset.toml"
    path.write_text(f'''schema_version = 1
dataset_name = "test dataset"
work_dir = "work/{{recording_session}}"
state_dir = "state"
steps = {json.dumps(steps)}
{extra}
[defaults]
schema_version = 8
lab = "test lab"
subject_id = "m1"
notes = "default"
[defaults.probes.imec0]
area = "V1"

[step_options.lfp]
n_jobs = 1
probes = ["imec0"]

[[recording_sessions]]
recording_session_id = "a"
recording_session_path = "{SPIKEGLX_CROP_RECORDING_SESSION.as_posix()}"

[[recording_sessions]]
recording_session_id = "b"
recording_session_path = "{SPIKEGLX_CROP_RECORDING_SESSION.as_posix()}"
recording_session_config_file = "recording_sessions/b.toml"
[recording_sessions.overrides]
task = "rf"
{overrides_b}
''')
    return path


def test_merge_tables():
    base = {"a": 1, "t": {"x": 1, "y": [1, 2]}, "runs": [{"name": "r1"}]}
    assert merge_tables(base, {"t": {"y": [3]}, "runs": [{"name": "r2"}]}) == \
        {"a": 1, "t": {"x": 1, "y": [3]}, "runs": [{"name": "r2"}]}
    assert base["t"]["y"] == [1, 2]  # not modified


def test_load_dataset(tmp_path):
    dataset = load_dataset(write_dataset(tmp_path))
    recording_session_a, recording_session_b = dataset.recording_sessions
    assert dataset.steps == STEPS and dataset.state_dir == tmp_path / "state"
    assert recording_session_a.work_dir == tmp_path / "work" / "a" and recording_session_a.recording_session_path == SPIKEGLX_CROP_RECORDING_SESSION.resolve()
    config_a, config_b = recording_session_a.recording_session_config, recording_session_b.recording_session_config
    assert (config_a.lab, config_a.notes, config_a.probes["imec0"].area, config_a.session_id) == \
        ("test lab", "default", "V1", "a")
    # defaults < file < inline overrides
    assert (config_b.lab, config_b.notes, config_b.probes["imec0"].area, config_b.task) == \
        ("test lab", "from the file", "V2", "rf")
    assert recording_session_b.config_files == [tmp_path / "dataset.toml", tmp_path / "recording_sessions" / "b.toml"]
    assert step_argv("lfp", recording_session_a.step_options["lfp"]) == ["--n-jobs", "1", "--probes", "imec0"]


def test_load_dataset_errors(tmp_path):
    path = write_dataset(tmp_path, extra='[step_options.mua]\nn_job = 2')
    with pytest.raises(ValueError, match="unknown options of mua: n_job"):
        load_dataset(path)
    path = write_dataset(tmp_path, extra='[step_options.lfp.x]\n', steps=["lfp", "timing"])
    with pytest.raises(ValueError, match="unknown steps timing"):
        load_dataset(path)
    path = write_dataset(tmp_path, extra='[step_options.export]\nwork_dir = "x"')
    with pytest.raises(ValueError, match="work_dir of export are set by the runner"):
        load_dataset(path)


def _states(dataset, recording_session_id):
    return {step: (read_record(dataset.recording_session(recording_session_id), step) or {}).get("state") for step in dataset.steps}


def test_run_resume_and_reruns(tmp_path):
    path = write_dataset(tmp_path)
    lines: list[str] = []
    results = Runner(load_dataset(path), echo=lines.append).run()
    assert results == {"a": "done", "b": "done"}, "\n".join(lines)
    dataset = load_dataset(path)
    assert _states(dataset, "a") == {s: "done" for s in STEPS}
    work = tmp_path / "work" / "b"
    assert (work / "lfp").is_dir() and (work / "events").is_dir() and (work / "clocks").is_dir()
    assert (work / "session.toml").read_text().count('area = "V2"') == 1  # the merged config
    first = {s: read_record(dataset.recording_session("b"), s)["run_id"] for s in STEPS}
    assert (tmp_path / "state" / "status.tsv").read_text().splitlines()[0].split("\t")[1:4] == STEPS

    # nothing to do
    lines.clear()
    assert Runner(load_dataset(path), echo=lines.append).run() == {"a": "done", "b": "done"}
    assert not [line for line in lines if line.split()[2:3] in (["a"], ["b"]) and line.split()[3] in STEPS], lines

    # a config change of session b: only b runs again, all its steps (they all read the config)
    write_dataset(tmp_path, overrides_b='notes = "changed"')
    dataset = load_dataset(path)
    table = status_table(dataset)
    assert table.splitlines()[1].split()[1:] == ["+", "+", "+"] and table.splitlines()[2].split()[1:] == ["~", "~", "~"]
    assert Runner(dataset, dry_run=True, echo=lines.append).run() == {"a": "done", "b": "done"}
    assert _states(dataset, "b") == {s: "done" for s in STEPS}  # dry run: nothing written
    lines.clear()
    Runner(dataset, echo=lines.append).run()
    assert sum(" b " in f" {line} " and "inputs changed" in line for line in lines) == 3, lines
    assert not any(line.split()[2:3] == ["a"] for line in lines if "inputs changed" in line)
    assert all(read_record(dataset.recording_session("b"), s)["run_id"] != first[s] for s in STEPS)

    # an upstream step that runs again makes the later ones run again
    record = read_record(dataset.recording_session("a"), "align-clocks")
    (tmp_path / "work" / "a" / "runner" / "align-clocks.json").write_text(json.dumps({**record, "state": "failed"}))
    assert status_table(dataset).splitlines()[1].split()[1:] == ["X", "~", "~"]

    # code changes count only without --ignore-code-changes
    dataset = load_dataset(path)
    for s in STEPS:
        f = tmp_path / "work" / "b" / "runner" / f"{s}.json"
        f.write_text(json.dumps({**json.loads(f.read_text()), "code": {"ephysalis": "0.0.1"}}))
    assert status_table(dataset).splitlines()[2].split()[1:] == ["c", "c", "c"]
    assert status_table(dataset, ignore_code_changes=True).splitlines()[2].split()[1:] == ["+", "+", "+"]


def test_pause_and_resume(tmp_path):
    path = write_dataset(tmp_path)
    dataset = load_dataset(path)

    def pause_after_first_step(line):
        if "a align-clocks" in line:  # the first step has started: ask for a pause
            request_pause(dataset)

    results = Runner(dataset, echo=pause_after_first_step).run(parallel=1)
    assert results == {"a": "paused", "b": "paused"}
    assert _states(dataset, "a") == {"align-clocks": "done", "extract-events": None, "lfp": None}
    assert _states(dataset, "b") == {s: None for s in STEPS}
    assert "paused" in status_table(dataset)
    assert main(["run", str(path)]) == 0  # resumes, removes the pause file
    assert not (tmp_path / "state" / PAUSE_FILE).exists()
    assert _states(load_dataset(path), "b") == {s: "done" for s in STEPS}


def test_pause_command(tmp_path):
    path = write_dataset(tmp_path)
    assert main(["pause", str(path)]) == 0
    assert (tmp_path / "state" / PAUSE_FILE).exists()
    assert main(["status", str(path)]) == 0


def test_lock_and_skipped_step(tmp_path):
    import os
    import socket

    path = write_dataset(tmp_path, extra='[step_options.sort]\nprobes = ["v4"]', steps=["align-clocks", "sort"])
    lock = tmp_path / "work" / "a" / "runner" / "lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))  # a live runner
    dataset = load_dataset(path)
    assert Runner(dataset, echo=lambda _: None).run() == {"a": "locked", "b": "done"}
    record = read_record(dataset.recording_session("b"), "sort")
    assert record["state"] == "skipped" and "v4" in record["reason"]
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": 2 ** 22 + 12345}))  # stale
    assert Runner(dataset, echo=lambda _: None).run(["a"]) == {"a": "done"}
    assert not lock.exists()


def test_source_data_are_never_written_or_deleted(tmp_path):
    """Work dirs inside a recording session folder are refused, and the runner deletes nothing outside the work
    dir and the sort scratch dir."""
    source = tmp_path / "source" / "ses-1"
    source.mkdir(parents=True)
    (tmp_path / "dataset.toml").write_text(
        'schema_version = 1\nwork_dir = "source/ses-1/ephysalis"\n[defaults]\nsubject_id = "m"\n'
        '[[recording_sessions]]\nrecording_session_id = "s"\nrecording_session_path = "source/ses-1"\n')
    with pytest.raises(ValueError, match="overlaps the recording session folder"):
        load_dataset(tmp_path / "dataset.toml")
    (tmp_path / "dataset.toml").write_text(
        'schema_version = 1\n[defaults]\nsubject_id = "m"\n'
        '[[recording_sessions]]\nrecording_session_id = "s"\nrecording_session_path = "source/ses-1"\n')
    with pytest.raises(ValueError, match="work_dir is required"):
        load_dataset(tmp_path / "dataset.toml")

    dataset = load_dataset(write_dataset(tmp_path))
    runner = Runner(dataset, echo=lambda _: None)
    session_a = dataset.recording_session("a")
    for outside in (tmp_path, session_a.recording_session_path, session_a.recording_session_path / "x",
                    session_a.work_dir):
        with pytest.raises(RuntimeError, match="refusing to delete"):
            runner._delete_tree(session_a, outside)
    inside = session_a.work_dir / "sorting" / "imec0"
    inside.mkdir(parents=True)
    runner._delete_tree(session_a, inside)
    assert not inside.exists() and session_a.recording_session_path.exists()
