"""``ephysalis run``: the steps of every session of a dataset (:mod:`ephysalis.dataset.config`), resumable.

Each session's state is in its work dir: ``runner/<step>.json`` (the step's last run: state, stamp,
command, log, duration), ``runner/logs/<step>.log`` and ``runner/lock`` while a runner works on it.
The dataset's state is in ``state_dir``: ``pause`` (see below) and ``status.tsv``.

A step is done when its record says ``done`` with the same stamp. The stamp has two parts:

- inputs: the session's merged config, the files it names (probe geometry), the step's options and
  the runs of the steps whose outputs it reads (:data:`~ephysalis.dataset.config.STEP_INPUTS`); so a
  step runs again when its config or options change or a step before it ran again;
- code: the versions of ephysalis (with its git commit) and of the plugin packages.
  ``--ignore-code-changes`` counts steps done with other code as done.

Steps that are not done run again from scratch (``--force``; ``sort`` with a new sorting), except a
``sort`` that was interrupted with the same stamp, which resumes. Steps run as subprocesses, so a
crashing step does not take the runner down; a failed step stops its session (the others go on).

Pausing: ``ephysalis pause`` writes ``state_dir/pause``; runners start no new step, the running steps
finish and the runners exit. ``ephysalis run`` again resumes (and removes the pause file). Ctrl+C
(SIGINT/SIGTERM) stops the running steps at once; they run again on resume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ephysalis.dataset.config import STEP_INPUTS, Dataset, DatasetRecordingSession, step_argv
from ephysalis.recording_session.config import save_recording_session_config

logger = logging.getLogger(__name__)
FORCE_STEPS = ("lfp", "csd", "mua")  # steps that keep existing outputs without --force (align-clocks: the sync
# edge cache only depends on the recording; the other steps overwrite their outputs)
PAUSE_FILE = "pause"
_STATUS_LOCK = threading.Lock()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_versions() -> dict[str, Any]:
    """Versions of ephysalis and the plugin packages, with their git commits when installed from checkouts."""
    from importlib import metadata

    from ephysalis.provenance import package_source, plugin_packages

    versions: dict[str, Any] = {}
    for package in ("ephysalis", *plugin_packages()):
        versions[package] = metadata.version(package)
        source = package_source(package)  # installed from a checkout: its commit (fixes without a new version)
        if source and source.get("commit"):
            versions[f"{package}_commit"] = source["commit"] + ("+dirty" if source.get("dirty") else "")
    return versions


def sort_scratch_dir(dataset: Dataset, dataset_recording_session: DatasetRecordingSession) -> Path:
    """The sort step's scratch dir of a recording session (its ``scratch_dir`` option; default: the system's)."""
    scratch = Path(dataset_recording_session.step_options.get("sort", {}).get("scratch_dir")
                   or Path(tempfile.gettempdir()) / "ephysalis")
    return scratch if scratch.is_absolute() else dataset.root / scratch


def runner_dir(dataset_recording_session: DatasetRecordingSession) -> Path:
    return dataset_recording_session.work_dir / "runner"


def read_record(dataset_recording_session: DatasetRecordingSession, step: str) -> dict[str, Any] | None:
    path = runner_dir(dataset_recording_session) / f"{step}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_record(dataset_recording_session: DatasetRecordingSession, step: str, record: dict[str, Any]) -> None:
    path = runner_dir(dataset_recording_session) / f"{step}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    tmp.replace(path)


def write_recording_session_config(dataset_recording_session: DatasetRecordingSession) -> Path:
    """The session's merged config as ``<work dir>/session.toml`` (rewritten only when it changed)."""
    path = dataset_recording_session.work_dir / "session.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "session.toml"
        save_recording_session_config(dataset_recording_session.recording_session_config, new)
        text = new.read_bytes()
    if not path.exists() or path.read_bytes() != text:
        path.write_bytes(text)
    return path


def step_inputs(dataset: Dataset, dataset_recording_session: DatasetRecordingSession, step: str, config_text: bytes) -> dict[str, Any]:
    """What the result of ``step`` depends on, apart from the code (see the module docstring)."""
    upstream = {}
    for dep in STEP_INPUTS[step]:
        if dep in dataset.steps:
            record = read_record(dataset_recording_session, dep)
            upstream[dep] = record.get("run_id") if record and record.get("state") in ("done", "skipped") else None
    files = {}
    for file in dataset_recording_session.referenced_files:
        files[str(file)] = _sha256(file.read_bytes()) if file.exists() else None
    return {"config": _sha256(config_text), "files": files,
            "options": step_argv(step, dataset_recording_session.step_options.get(step, {})), "upstream": upstream}


@dataclass
class StepState:
    state: str  # done, skipped, pending (never ran), changed (inputs), code (code changed), failed, interrupted, running
    record: dict[str, Any] | None

    @property
    def finished(self) -> bool:
        return self.state in ("done", "skipped")


def step_state(dataset: Dataset, dataset_recording_session: DatasetRecordingSession, step: str, code: dict[str, Any], config_text: bytes,
               ignore_code_changes: bool = False) -> StepState:
    record = read_record(dataset_recording_session, step)
    if record is None:
        return StepState("pending", None)
    if record.get("state") == "running":
        return StepState("running" if _lock_alive(dataset_recording_session) else "interrupted", record)
    if record.get("state") in ("failed", "interrupted"):
        return StepState(record["state"], record)
    if record.get("inputs") != step_inputs(dataset, dataset_recording_session, step, config_text):
        return StepState("changed", record)
    if record.get("code") != code and not ignore_code_changes:
        return StepState("code", record)
    return StepState(record.get("state", "pending"), record)


# --- locks ---------------------------------------------------------------------------------------------------------

def _lock_path(dataset_recording_session: DatasetRecordingSession) -> Path:
    return runner_dir(dataset_recording_session) / "lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:  # Windows: no such process
        return False
    return True


def _lock_owner(dataset_recording_session: DatasetRecordingSession) -> dict[str, Any] | None:
    try:
        return json.loads(_lock_path(dataset_recording_session).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _lock_alive(dataset_recording_session: DatasetRecordingSession) -> bool:
    owner = _lock_owner(dataset_recording_session)
    if owner is None:
        return False
    if owner.get("host") != socket.gethostname():
        return True  # another machine's runner: cannot check, take it as alive
    return _pid_alive(int(owner.get("pid", -1)))


def acquire_lock(dataset_recording_session: DatasetRecordingSession) -> bool:
    """Take the session's lock (False: another runner holds it). Stale locks of this host are taken over."""
    path = _lock_path(dataset_recording_session)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                        "started": datetime.now(UTC).isoformat(timespec="seconds")})
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_alive(dataset_recording_session):
                return False
            path.unlink(missing_ok=True)  # stale: its runner is gone
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(owner)
        return True
    return False


def release_lock(dataset_recording_session: DatasetRecordingSession) -> None:
    owner = _lock_owner(dataset_recording_session)
    if owner and owner.get("pid") == os.getpid() and owner.get("host") == socket.gethostname():
        _lock_path(dataset_recording_session).unlink(missing_ok=True)


# --- running ------------------------------------------------------------------------------------------------------

def pause_requested(dataset: Dataset) -> bool:
    return (dataset.state_dir / PAUSE_FILE).exists()


def request_pause(dataset: Dataset) -> Path:
    dataset.state_dir.mkdir(parents=True, exist_ok=True)
    path = dataset.state_dir / PAUSE_FILE
    path.write_text(datetime.now(UTC).isoformat(timespec="seconds") + "\n", encoding="utf-8")
    return path


class Runner:
    def __init__(self, dataset: Dataset, steps: list[str] | None = None, ignore_code_changes: bool = False,
                 dry_run: bool = False, echo=None):
        self.dataset = dataset
        self.steps = [s for s in dataset.steps if steps is None or s in steps]
        self.ignore_code_changes = ignore_code_changes
        self.dry_run = dry_run
        self.echo = echo or (lambda text: print(text, flush=True))  # flushed: the log of a detached runner
        self.code = code_versions()
        self.stop = threading.Event()
        self._children: set[subprocess.Popen[bytes]] = set()
        self._children_lock = threading.Lock()

    def _say(self, text: str) -> None:
        self.echo(f"{datetime.now():%m-%d %H:%M:%S} {text}")

    def interrupt(self) -> None:
        """Stop at once: terminate the running steps (Ctrl+C)."""
        self.stop.set()
        with self._children_lock:
            for child in self._children:
                child.terminate()

    def _skip_reason(self, dataset_recording_session: DatasetRecordingSession, step: str) -> str | None:
        probes = dataset_recording_session.step_options.get(step, {}).get("probes")
        if probes and not set(probes) & set(dataset_recording_session.recording_session_config.probes):
            return f"none of the probes {', '.join(probes)} in the session config"
        return None

    def _fresh(self, dataset_recording_session: DatasetRecordingSession, step: str, previous: StepState, inputs: dict[str, Any],
               record: dict[str, Any] | None) -> None:
        """Remove what a step would reuse, unless it resumes an interrupted run with the same stamp."""
        if step != "sort":
            return
        resume = (previous.state in ("interrupted", "failed") and record is not None
                  and record.get("inputs") == inputs and record.get("code") == self.code)
        if resume:
            self._say(f"{dataset_recording_session.recording_session_id} sort: resuming the interrupted sorting")
            return
        probes = (dataset_recording_session.step_options.get("sort", {}).get("probes")
                  or list(dataset_recording_session.recording_session_config.probes))
        scratch = sort_scratch_dir(self.dataset, dataset_recording_session)
        for probe in probes:
            self._delete_tree(dataset_recording_session, dataset_recording_session.work_dir / "sorting" / probe)
            for old in scratch.glob(f"{dataset_recording_session.recording_session_path.name}_{probe}*"):
                self._delete_tree(dataset_recording_session, old)

    def _delete_tree(self, dataset_recording_session: DatasetRecordingSession, folder: Path) -> None:
        """Delete ``folder`` only if it lies in the recording session's work dir or the sort scratch dir, and in no
        recording session folder (the source data are never touched)."""
        target = folder.resolve()
        allowed = [dataset_recording_session.work_dir.resolve(),
                   sort_scratch_dir(self.dataset, dataset_recording_session).resolve()]
        protected = [s.recording_session_path.resolve() for s in self.dataset.recording_sessions]
        if (not any(target != root and target.is_relative_to(root) for root in allowed)
                or any(target == source or target.is_relative_to(source) or source.is_relative_to(target)
                       for source in protected)):
            raise RuntimeError(f"refusing to delete {target}: not inside the work dir or the sort scratch dir of "
                               f"{dataset_recording_session.recording_session_id}, or it holds source data")
        if target.exists():
            shutil.rmtree(target)

    def run_step(self, dataset_recording_session: DatasetRecordingSession, step: str, config_path: Path, config_text: bytes) -> bool:
        previous = step_state(self.dataset, dataset_recording_session, step, self.code, config_text, self.ignore_code_changes)
        if previous.finished:
            return True
        inputs = step_inputs(self.dataset, dataset_recording_session, step, config_text)
        base = {"step": step, "inputs": inputs, "code": self.code, "run_id": uuid.uuid4().hex,
                "started": datetime.now(UTC).isoformat(timespec="seconds"), "host": socket.gethostname()}
        if reason := self._skip_reason(dataset_recording_session, step):
            if not self.dry_run:
                _write_record(dataset_recording_session, step, {**base, "state": "skipped", "reason": reason})
            self._say(f"{dataset_recording_session.recording_session_id} {step}: skipped ({reason})")
            return True
        argv = [sys.executable, "-u", "-m", "ephysalis.cli.main", step, str(dataset_recording_session.recording_session_path),
                "--config", str(config_path), "--work-dir", str(dataset_recording_session.work_dir),
                *step_argv(step, dataset_recording_session.step_options.get(step, {}))]
        if step in FORCE_STEPS:
            argv.append("--force")
        why = {"pending": "", "changed": " (inputs changed)", "code": " (code changed)", "failed": " (failed before)",
               "interrupted": " (interrupted before)"}.get(previous.state, "")
        if self.dry_run:
            self._say(f"{dataset_recording_session.recording_session_id} {step}: would run{why}")
            return True
        self._fresh(dataset_recording_session, step, previous, inputs, previous.record)
        log = runner_dir(dataset_recording_session) / "logs" / f"{step}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        record = {**base, "state": "running", "argv": argv[3:], "log": str(log), "pid": os.getpid()}
        _write_record(dataset_recording_session, step, record)
        self._say(f"{dataset_recording_session.recording_session_id} {step}{why}")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as out:
            child = subprocess.Popen(argv, cwd=self.dataset.root, stdout=out, stderr=subprocess.STDOUT)
            with self._children_lock:
                self._children.add(child)
            try:
                code = child.wait()
            finally:
                with self._children_lock:
                    self._children.discard(child)
        seconds = round(time.time() - t0)
        if self.stop.is_set() and code != 0:
            _write_record(dataset_recording_session, step, {**record, "state": "interrupted", "seconds": seconds})
            self._say(f"{dataset_recording_session.recording_session_id} {step}: interrupted")
            return False
        state = "done" if code == 0 else "failed"
        _write_record(dataset_recording_session, step, {**record, "state": state, "exit_code": code, "seconds": seconds,
                                      "finished": datetime.now(UTC).isoformat(timespec="seconds")})
        if code != 0:
            self._say(f"{dataset_recording_session.recording_session_id} {step}: failed (exit {code}), see {log}")
        return code == 0

    def run_recording_session(self, dataset_recording_session: DatasetRecordingSession) -> str:
        """``done``, ``failed``, ``paused``, ``interrupted`` or ``locked``."""
        if self.stop.is_set():
            return "interrupted"
        if not self.dry_run and not acquire_lock(dataset_recording_session):
            owner = _lock_owner(dataset_recording_session) or {}
            self._say(f"{dataset_recording_session.recording_session_id}: another runner works on it ({owner.get('host')}, pid {owner.get('pid')})")
            return "locked"
        try:
            config_path = write_recording_session_config(dataset_recording_session) if not self.dry_run else dataset_recording_session.work_dir / "session.toml"
            with tempfile.TemporaryDirectory() as tmp:
                save_recording_session_config(dataset_recording_session.recording_session_config, Path(tmp) / "s.toml")
                config_text = (Path(tmp) / "s.toml").read_bytes()
            for step in self.steps:
                if self.stop.is_set():
                    return "interrupted"
                if pause_requested(self.dataset):
                    return "paused"
                if not self.run_step(dataset_recording_session, step, config_path, config_text):
                    return "interrupted" if self.stop.is_set() else "failed"
            return "done"
        finally:
            if not self.dry_run:
                release_lock(dataset_recording_session)

    def run(self, recording_session_ids: list[str] | None = None, parallel: int = 1) -> dict[str, str]:
        dataset_recording_sessions = [dataset_recording_session for dataset_recording_session in self.dataset.recording_sessions if dataset_recording_session.include
                            and (recording_session_ids is None or dataset_recording_session.recording_session_id in recording_session_ids)]
        if recording_session_ids:
            missing = set(recording_session_ids) - {dataset_recording_session.recording_session_id for dataset_recording_session in self.dataset.recording_sessions}
            if missing:
                raise KeyError(f"not in {self.dataset.dataset_file}: {', '.join(sorted(missing))}")
        if not self.dry_run:
            (self.dataset.state_dir / PAUSE_FILE).unlink(missing_ok=True)  # resume
        self._say(f"{self.dataset.dataset_name}: {len(dataset_recording_sessions)} sessions, {parallel} at a time, steps {', '.join(self.steps)}")
        results: dict[str, str] = {}

        def run_one_recording_session(dataset_recording_session: DatasetRecordingSession) -> None:
            t0 = time.time()
            result = self.run_recording_session(dataset_recording_session)
            results[dataset_recording_session.recording_session_id] = result
            if self.dry_run:
                return
            if result not in ("paused", "interrupted", "locked") or time.time() - t0 > 1:
                self._say(f"{dataset_recording_session.recording_session_id}: {result} ({(time.time() - t0) / 60:.0f} min)")
            if not self.dry_run:
                write_status(self.dataset, [dataset_recording_session], self.code, self.ignore_code_changes)

        with ThreadPoolExecutor(max(1, parallel)) as pool:
            list(pool.map(run_one_recording_session, dataset_recording_sessions))
        if self.dry_run:
            self._say("dry run: nothing was run")
            return results
        result_counts = {result: list(results.values()).count(result) for result in sorted(set(results.values()))}
        self._say(", ".join(f"{count} {result}" for result, count in result_counts.items()) or "no sessions")
        if pause_requested(self.dataset):
            self._say(f"paused; resume with: ephysalis run {self.dataset.dataset_file}")
        return results


# --- status -------------------------------------------------------------------------------------------------------

SYMBOLS = {"done": "+", "skipped": "-", "pending": ".", "changed": "~", "code": "c", "failed": "X",
           "interrupted": "!", "running": ">"}


def recording_session_states(dataset: Dataset, dataset_recording_session: DatasetRecordingSession, code: dict[str, Any],
                   ignore_code_changes: bool = False) -> dict[str, StepState]:
    with tempfile.TemporaryDirectory() as tmp:
        save_recording_session_config(dataset_recording_session.recording_session_config, Path(tmp) / "s.toml")
        config_text = (Path(tmp) / "s.toml").read_bytes()
    states: dict[str, StepState] = {}
    for step in dataset.steps:
        state = step_state(dataset, dataset_recording_session, step, code, config_text, ignore_code_changes)
        if state.finished and any(dep in states and not states[dep].finished for dep in STEP_INPUTS[step]):
            state = StepState("changed", state.record)  # runs again after the step before it
        states[step] = state
    return states


def status_table(dataset: Dataset, code: dict[str, Any] | None = None, ignore_code_changes: bool = False,
                 recording_session_ids: list[str] | None = None) -> str:
    code = code if code is not None else code_versions()
    width = max([len(dataset_recording_session.recording_session_id) for dataset_recording_session in dataset.recording_sessions] + [7])
    step_columns = [step.replace("extract-", "x-").replace("align-clocks", "clocks") for step in dataset.steps]
    lines = [f"{'recording_session':{width}}  " + " ".join(step_columns)]
    for dataset_recording_session in dataset.recording_sessions:
        if recording_session_ids and dataset_recording_session.recording_session_id not in recording_session_ids:
            continue
        if not dataset_recording_session.include:
            lines.append(f"{dataset_recording_session.recording_session_id:{width}}  (excluded)")
            continue
        states = recording_session_states(dataset, dataset_recording_session, code, ignore_code_changes)
        cells = [f"{SYMBOLS[states[step].state]:^{len(column)}}" for step, column in zip(dataset.steps, step_columns)]
        lines.append(f"{dataset_recording_session.recording_session_id:{width}}  " + " ".join(cells))
    lines.append("+ done  - skipped  . not run  ~ inputs changed  c code changed  X failed  ! interrupted  > running")
    if pause_requested(dataset):
        lines.append(f"paused ({dataset.state_dir / PAUSE_FILE})")
    return "\n".join(lines)


def write_status(dataset: Dataset, dataset_recording_sessions: list[DatasetRecordingSession], code: dict[str, Any],
                 ignore_code_changes: bool = False) -> None:
    """Update ``state_dir/status.tsv`` (session x step states) for ``dataset_sessions``."""
    path = dataset.state_dir / "status.tsv"
    with _STATUS_LOCK:
        rows: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines()[1:]:
                rows[line.split("\t", 1)[0]] = line
        for dataset_recording_session in dataset_recording_sessions:
            states = recording_session_states(dataset, dataset_recording_session, code, ignore_code_changes)
            rows[dataset_recording_session.recording_session_id] = "\t".join([dataset_recording_session.recording_session_id,
                                                          *(states[step].state for step in dataset.steps),
                                                          datetime.now().isoformat(timespec="seconds")])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\t".join(["recording_session", *dataset.steps, "updated"]) + "\n"
                        + "\n".join(rows[k] for k in sorted(rows)) + "\n", encoding="utf-8")


def install_signal_handlers(runner: Runner) -> None:
    def handler(signum, frame):
        runner._say("stopping: terminating the running steps (they run again on resume)")
        runner.interrupt()

    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)
