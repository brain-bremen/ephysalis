"""conversion_batch.toml (defaults, per-session override files and inline overrides) and ``ephysalis run``: steps
done once, reruns after config changes of only the affected session, pause and resume, locks, skipped
steps, code changes."""

import json
from pathlib import Path

import pytest
from conftest import SPIKEGLX_CROP_RECORDING_SESSION

from ephysalis.cli.main import main
from ephysalis.conversion_batch.config import load_conversion_batch, merge_tables, step_argv
from ephysalis.conversion_batch.runner import (
    PAUSE_FILE,
    Runner,
    read_record,
    request_pause,
    status_table,
)

STEPS = ["align-clocks", "extract-events", "lfp"]


def write_conversion_batch(root: Path, overrides_b: str = "", extra: str = "", steps=STEPS) -> Path:
    (root / "session_configs").mkdir(exist_ok=True)
    (root / "session_configs" / "b.toml").write_text('schema_version = 10\nnotes = "from the file"\n'
                                                     '[probes.imec0]\narea = "V2"\n')
    for name in ("a", "b"):  # two recording session folders with the same recording
        if not (root / "sessions" / name).exists():
            (root / "sessions").mkdir(exist_ok=True)
            (root / "sessions" / name).symlink_to(SPIKEGLX_CROP_RECORDING_SESSION.resolve(), target_is_directory=True)
    session_config_b = 'task = "rf"' + (f", {overrides_b}" if overrides_b else "")
    path = root / "conversion_batch.toml"
    path.write_text(f'''schema_version = 2
conversion_batch_name = "test conversion batch"
work_dir = "work/{{recording_session}}"
state_dir = "state"
steps = {json.dumps(steps)}
recording_session_root = "sessions"
recording_sessions = [
    {{ subject = "m1", folder = "a" }},
    {{ subject = "m1", folder = "b", recording_session_config_file = "session_configs/b.toml", session_config = {{ {session_config_b} }} }},
]
{extra}
[defaults]
schema_version = 10
lab = "test lab"
notes = "default"
[defaults.probes.imec0]
area = "V1"

[step_options.lfp]
n_jobs = 1
probes = ["imec0"]

''')
    return path


def test_merge_tables():
    base = {"a": 1, "t": {"x": 1, "y": [1, 2]}, "runs": [{"name": "r1"}]}
    assert merge_tables(base, {"t": {"y": [3]}, "runs": [{"name": "r2"}]}) == \
        {"a": 1, "t": {"x": 1, "y": [3]}, "runs": [{"name": "r2"}]}
    assert base["t"]["y"] == [1, 2]  # not modified


def test_load_conversion_batch(tmp_path):
    conversion_batch = load_conversion_batch(write_conversion_batch(tmp_path))
    recording_session_a, recording_session_b = conversion_batch.recording_session_jobs
    assert conversion_batch.steps == STEPS and conversion_batch.state_dir == tmp_path / "state"
    assert recording_session_a.work_dir == tmp_path / "work" / "a" and recording_session_a.recording_session_path == SPIKEGLX_CROP_RECORDING_SESSION.resolve()
    config_a, config_b = recording_session_a.recording_session_config, recording_session_b.recording_session_config
    assert (config_a.lab, config_a.notes, config_a.probes["imec0"].area, config_a.session_id) == \
        ("test lab", "default", "V1", "a")
    # defaults < file < inline overrides
    assert (config_b.lab, config_b.notes, config_b.probes["imec0"].area, config_b.task) == \
        ("test lab", "from the file", "V2", "rf")
    assert recording_session_b.config_files == [tmp_path / "conversion_batch.toml", tmp_path / "session_configs" / "b.toml"]
    assert step_argv("lfp", recording_session_a.step_options["lfp"]) == ["--n-jobs", "1", "--probes", "imec0"]


def test_recording_sessions_of_several_subjects(tmp_path):
    """recording_sessions = [{ subject, folder, ... }]: id = folder with _, subject_id = subject, defaults per subject,
    placeholders {recording_session}, {subject}, {session_label} in the config file template and step options."""
    for subject, day in (("b", "20101207"), ("b", "20101209"), ("f", "20091203")):
        (tmp_path / "sourcedata" / f"sub-{subject}" / f"ses-{day}").mkdir(parents=True)
        (tmp_path / "sessions" / f"sub-{subject}_ses-{day}").mkdir(parents=True)
        (tmp_path / "sessions" / f"sub-{subject}_ses-{day}" / "session.toml").write_text(f'notes = "{day}"\n')
    path = tmp_path / "conversion_batch.toml"
    path.write_text('''schema_version = 2
work_dir = "work/{recording_session}"
recording_session_root = "sourcedata"
recording_session_config_file = "sessions/{recording_session}/session.toml"
recording_sessions = [
    { subject = "b", folder = "sub-b/ses-20101207" },
    { subject = "b", folder = "sub-b/ses-20101209", include = false, step_options = { lfp = { n_jobs = 2 } } },
    { subject = "f", folder = "sub-f/ses-20091203" },
]
[defaults]
schema_version = 10
lab = "lab"
[subject_defaults.b]
experimenter = ["I. G."]
[subject_defaults.b.aux_channels.digital]
trial_start = { bit = 0, description = "trial start" }
[subject_defaults.f]
experimenter = ["S. N."]
[step_options.lfp]
n_jobs = 1
probes = ["{subject}_{session_label}"]
''')
    first, second, third = load_conversion_batch(path).recording_session_jobs
    assert [j.recording_session_id for j in (first, second, third)] == [
        "sub-b_ses-20101207", "sub-b_ses-20101209", "sub-f_ses-20091203"]
    assert first.recording_session_path == tmp_path / "sourcedata" / "sub-b" / "ses-20101207"
    assert first.work_dir == tmp_path / "work" / "sub-b_ses-20101207" and first.include and not second.include
    a, c = first.recording_session_config, third.recording_session_config
    assert (a.subject_id, a.session_id, a.notes, a.lab, a.experimenter) == (
        "b", "sub-b_ses-20101207", "20101207", "lab", ["I. G."])
    assert a.aux_channels.digital["trial_start"].description == "trial start"
    assert (c.subject_id, c.experimenter) == ("f", ["S. N."]) and "trial_start" not in c.aux_channels.digital
    assert first.step_options["lfp"] == {"n_jobs": 1, "probes": ["b_20101207"]}
    assert second.step_options["lfp"]["n_jobs"] == 2

    text = path.read_text()
    path.write_text(text.replace('"sub-f/ses-20091203" }', '"sub-f/ses-20091204" }'))
    with pytest.raises(ValueError, match="config file .* not found"):
        load_conversion_batch(path)
    path.write_text(text.replace('"sub-b/ses-20101209", include', '"sub-b/ses-20101207", include'))
    with pytest.raises(ValueError, match="listed twice .*: sub-b_ses-20101207"):
        load_conversion_batch(path)
    path.write_text(text.replace('{ subject = "f", folder = "sub-f/ses-20091203" },', ""))
    with pytest.raises(ValueError, match="subject_defaults of subjects without recording sessions: f"):
        load_conversion_batch(path)
    path.write_text(text.replace('lab = "lab"', 'lab = "lab"\nsubject_id = "b"'))
    with pytest.raises(ValueError, match="sets subject_id"):
        load_conversion_batch(path)
    (tmp_path / "sessions" / "sub-f_ses-20091203" / "session.toml").write_text('subject_id = "x"\n')
    path.write_text(text)
    with pytest.raises(ValueError, match="subject_id 'x' of its config is not the subject 'f'"):
        load_conversion_batch(path)


def test_load_dataset_errors(tmp_path):
    path = write_conversion_batch(tmp_path, extra='[step_options.mua]\nn_job = 2')
    with pytest.raises(ValueError, match="unknown options of mua: n_job"):
        load_conversion_batch(path)
    path = write_conversion_batch(tmp_path, extra='[step_options.lfp.x]\n', steps=["lfp", "timing"])
    with pytest.raises(ValueError, match="unknown steps timing"):
        load_conversion_batch(path)
    path = write_conversion_batch(tmp_path, extra='[step_options.export]\nwork_dir = "x"')
    with pytest.raises(ValueError, match="work_dir of export are set by the runner"):
        load_conversion_batch(path)


def _states(conversion_batch, recording_session_id):
    return {step: (read_record(conversion_batch.recording_session_job(recording_session_id), step) or {}).get("state") for step in conversion_batch.steps}


def test_run_resume_and_reruns(tmp_path):
    path = write_conversion_batch(tmp_path)
    lines: list[str] = []
    results = Runner(load_conversion_batch(path), echo=lines.append).run()
    assert results == {"a": "done", "b": "done"}, "\n".join(lines)
    conversion_batch = load_conversion_batch(path)
    assert _states(conversion_batch, "a") == {s: "done" for s in STEPS}
    work = tmp_path / "work" / "b"
    assert (work / "lfp").is_dir() and (work / "events").is_dir() and (work / "clocks").is_dir()
    assert (work / "session.toml").read_text().count('area = "V2"') == 1  # the merged config
    first = {s: read_record(conversion_batch.recording_session_job("b"), s)["run_id"] for s in STEPS}
    assert (tmp_path / "state" / "status.tsv").read_text().splitlines()[0].split("\t")[1:4] == STEPS

    # nothing to do
    lines.clear()
    assert Runner(load_conversion_batch(path), echo=lines.append).run() == {"a": "done", "b": "done"}
    assert not [line for line in lines if line.split()[2:3] in (["a"], ["b"]) and line.split()[3] in STEPS], lines

    # a config change of session b: only b runs again, all its steps (they all read the config)
    write_conversion_batch(tmp_path, overrides_b='notes = "changed"')
    conversion_batch = load_conversion_batch(path)
    table = status_table(conversion_batch)
    assert table.splitlines()[1].split()[1:] == ["+", "+", "+"] and table.splitlines()[2].split()[1:] == ["~", "~", "~"]
    assert Runner(conversion_batch, dry_run=True, echo=lines.append).run() == {"a": "done", "b": "done"}
    assert _states(conversion_batch, "b") == {s: "done" for s in STEPS}  # dry run: nothing written
    lines.clear()
    Runner(conversion_batch, echo=lines.append).run()
    assert sum(" b " in f" {line} " and "inputs changed" in line for line in lines) == 3, lines
    assert not any(line.split()[2:3] == ["a"] for line in lines if "inputs changed" in line)
    assert all(read_record(conversion_batch.recording_session_job("b"), s)["run_id"] != first[s] for s in STEPS)

    # an upstream step that runs again makes the later ones run again
    record = read_record(conversion_batch.recording_session_job("a"), "align-clocks")
    (tmp_path / "work" / "a" / "runner" / "align-clocks.json").write_text(json.dumps({**record, "state": "failed"}))
    assert status_table(conversion_batch).splitlines()[1].split()[1:] == ["X", "~", "~"]

    # code changes count only without --ignore-code-changes
    conversion_batch = load_conversion_batch(path)
    for s in STEPS:
        f = tmp_path / "work" / "b" / "runner" / f"{s}.json"
        f.write_text(json.dumps({**json.loads(f.read_text()), "code": {"ephysalis": "0.0.1"}}))
    assert status_table(conversion_batch).splitlines()[2].split()[1:] == ["c", "c", "c"]
    assert status_table(conversion_batch, ignore_code_changes=True).splitlines()[2].split()[1:] == ["+", "+", "+"]


def test_pause_and_resume(tmp_path):
    path = write_conversion_batch(tmp_path)
    conversion_batch = load_conversion_batch(path)

    def pause_after_first_step(line):
        if "a align-clocks" in line:  # the first step has started: ask for a pause
            request_pause(conversion_batch)

    results = Runner(conversion_batch, echo=pause_after_first_step).run(parallel=1)
    assert results == {"a": "paused", "b": "paused"}
    assert _states(conversion_batch, "a") == {"align-clocks": "done", "extract-events": None, "lfp": None}
    assert _states(conversion_batch, "b") == {s: None for s in STEPS}
    assert "paused" in status_table(conversion_batch)
    assert main(["run", str(path)]) == 0  # resumes, removes the pause file
    assert not (tmp_path / "state" / PAUSE_FILE).exists()
    assert _states(load_conversion_batch(path), "b") == {s: "done" for s in STEPS}


def test_pause_command(tmp_path):
    path = write_conversion_batch(tmp_path)
    assert main(["pause", str(path)]) == 0
    assert (tmp_path / "state" / PAUSE_FILE).exists()
    assert main(["status", str(path)]) == 0


def test_lock_and_skipped_step(tmp_path):
    import os
    import socket

    path = write_conversion_batch(tmp_path, extra='[step_options.spike-sort]\nprobes = ["v4"]', steps=["align-clocks", "spike-sort"])
    lock = tmp_path / "work" / "a" / "runner" / "lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))  # a live runner
    conversion_batch = load_conversion_batch(path)
    assert Runner(conversion_batch, echo=lambda _: None).run() == {"a": "locked", "b": "done"}
    record = read_record(conversion_batch.recording_session_job("b"), "spike-sort")
    assert record["state"] == "skipped" and "v4" in record["reason"]
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": 2 ** 22 + 12345}))  # stale
    assert Runner(conversion_batch, echo=lambda _: None).run(["a"]) == {"a": "done"}
    assert not lock.exists()


def test_source_data_are_never_written_or_deleted(tmp_path):
    """Work dirs inside a recording session folder are refused, and the runner deletes nothing outside the work
    dir and the sort scratch dir."""
    source = tmp_path / "source" / "ses-1"
    source.mkdir(parents=True)
    (tmp_path / "conversion_batch.toml").write_text(
        'schema_version = 2\nwork_dir = "source/ses-1/ephysalis"\n'
        'recording_sessions = [{ subject = "m", folder = "source/ses-1" }]\n')
    with pytest.raises(ValueError, match="overlaps the recording session folder"):
        load_conversion_batch(tmp_path / "conversion_batch.toml")
    (tmp_path / "conversion_batch.toml").write_text(
        'schema_version = 2\n'
        'recording_sessions = [{ subject = "m", folder = "source/ses-1" }]\n')
    with pytest.raises(ValueError, match="work_dir is required"):
        load_conversion_batch(tmp_path / "conversion_batch.toml")

    conversion_batch = load_conversion_batch(write_conversion_batch(tmp_path))
    runner = Runner(conversion_batch, echo=lambda _: None)
    session_a = conversion_batch.recording_session_job("a")
    for outside in (tmp_path, session_a.recording_session_path, session_a.recording_session_path / "x",
                    session_a.work_dir):
        with pytest.raises(RuntimeError, match="refusing to delete"):
            runner._delete_tree(session_a, outside)
    inside = session_a.work_dir / "spike_sorting" / "imec0"
    inside.mkdir(parents=True)
    runner._delete_tree(session_a, inside)
    assert not inside.exists() and session_a.recording_session_path.exists()


def test_remove_folder(tmp_path):
    """remove_folder lists each directory completely before deleting (NFS servers whose listing skips entries)."""
    from ephysalis.filesystem import remove_folder

    for k in range(3):
        sub = tmp_path / "tree" / f"d{k}" / "e"
        sub.mkdir(parents=True)
        for n in range(50):
            (sub / f"f{n}.npy").write_bytes(b"x")
    (tmp_path / "file").write_bytes(b"x")
    remove_folder(tmp_path / "tree")
    remove_folder(tmp_path / "missing")
    remove_folder(tmp_path / "file")
    assert not (tmp_path / "tree").exists() and not (tmp_path / "file").exists()
