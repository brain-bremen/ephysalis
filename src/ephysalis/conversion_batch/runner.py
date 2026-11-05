"""``ephysalis run``: the steps of every session of a conversion batch (:mod:`ephysalis.conversion_batch.config`), resumable.

Each session's state is in its work dir: ``runner/<step>.json`` (the step's last run: state, stamp,
command, log, duration), ``runner/logs/<step>.log`` and ``runner/lock`` while a runner works on it.
The conversion batch's state is in ``state_dir``: ``pause`` (see below) and ``status.tsv``.

A step is done when its record says ``done`` with the same stamp. The stamp has two parts:

- inputs: the session's merged config, the files it names (probe geometry), the step's options and
  the runs of the steps whose outputs it reads (:data:`~ephysalis.conversion_batch.config.STEP_INPUTS`); so a
  step runs again when its config or options change or a step before it ran again;
- code: the versions of ephysalis (with its git commit) and of the plugin packages.
  ``--ignore-code-changes`` counts steps done with other code as done.

Steps that are not done run again from scratch (``--force``; ``sort`` with a new sorting), except a
``sort`` that was interrupted with the same stamp, which resumes. Steps run as subprocesses, so a
crashing step does not take the runner down; a failed step stops its session (the others go on).

Reading: ``[source_cache]`` (:mod:`ephysalis.conversion_batch.source_cache`) lets the steps read the session
folders directly or a copy on a local disk made before a session's first step that has to run. While sessions
compute, a prefetch thread copies the next ones (at most ``parallel`` copies ahead). In ``local`` mode the steps
also work in a local copy of the session's work dir, copied back when the session ends; the session's lock stays
in its work dir.

Pausing: ``ephysalis pause`` writes ``state_dir/pause``; runners start no new step, the running steps
finish and the runners exit. ``ephysalis run`` again resumes (and removes the pause file). Ctrl+C
(SIGINT/SIGTERM) stops the running steps at once; they run again on resume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from ephysalis.conversion_batch.config import (
    STEP_INPUTS,
    ConversionBatch,
    RecordingSessionJob,
)
from ephysalis.conversion_batch.source_cache import (
    ORIGIN_ENV,
    WORK_DIR_ORIGIN_ENV,
    SourceCache,
    SourceCacheSettings,
    Staged,
    mirror,
)
from ephysalis.filesystem import glob_in_work_folder, remove_folder
from ephysalis.step_settings import step_settings_argv
from ephysalis.recording_session.config import save_recording_session_config

logger = logging.getLogger(__name__)
FORCE_STEPS = ("lfp", "csd", "mua")  # steps that keep existing outputs without --force (align-clocks: the sync
# edge cache only depends on the recording; the other steps overwrite their outputs)
PAUSE_FILE = "pause"
PREFETCH_RETRY_SECONDS = 30.0  # a prefetch that does not fit yet tries again after this
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


def sort_scratch_dir(conversion_batch: ConversionBatch, recording_session_job: RecordingSessionJob) -> Path:
    """The sort step's scratch dir of a recording session (its ``scratch_dir`` option; default: the system's)."""
    scratch = Path(recording_session_job.step_settings.get("spike-sort", {}).get("scratch_dir")
                   or Path(tempfile.gettempdir()) / "ephysalis")
    return scratch if scratch.is_absolute() else conversion_batch.root / scratch


def runner_dir(recording_session_job: RecordingSessionJob) -> Path:
    return recording_session_job.work_dir / "runner"


def read_record(recording_session_job: RecordingSessionJob, step: str) -> dict[str, Any] | None:
    path = runner_dir(recording_session_job) / f"{step}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_record(recording_session_job: RecordingSessionJob, step: str, record: dict[str, Any]) -> None:
    path = runner_dir(recording_session_job) / f"{step}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    tmp.replace(path)


def write_recording_session_config(recording_session_job: RecordingSessionJob) -> Path:
    """The session's merged config as ``<work dir>/session.toml`` (rewritten only when it changed)."""
    path = recording_session_job.work_dir / "session.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "session.toml"
        save_recording_session_config(recording_session_job.recording_session_config, new)
        text = new.read_bytes()
    if not path.exists() or path.read_bytes() != text:
        path.write_bytes(text)
    return path


def step_inputs(conversion_batch: ConversionBatch, recording_session_job: RecordingSessionJob, step: str, config_text: bytes) -> dict[str, Any]:
    """What the result of ``step`` depends on, apart from the code (see the module docstring)."""
    upstream = {}
    for dep in STEP_INPUTS[step]:
        if dep in conversion_batch.steps:
            record = read_record(recording_session_job, dep)
            upstream[dep] = record.get("run_id") if record and record.get("state") in ("done", "skipped") else None
    files = {}
    for file in recording_session_job.referenced_files:
        files[str(file)] = _sha256(file.read_bytes()) if file.exists() else None
    return {"config": _sha256(config_text), "files": files,
            "options": step_settings_argv(step, recording_session_job.step_settings[step]), "upstream": upstream}


@dataclass
class StepState:
    state: str  # done, skipped, pending (never ran), changed (inputs), code (code changed), failed, interrupted, running
    record: dict[str, Any] | None

    @property
    def finished(self) -> bool:
        return self.state in ("done", "skipped")


def step_state(conversion_batch: ConversionBatch, recording_session_job: RecordingSessionJob, step: str, code: dict[str, Any], config_text: bytes,
               ignore_code_changes: bool = False) -> StepState:
    record = read_record(recording_session_job, step)
    if record is None:
        return StepState("pending", None)
    if record.get("state") == "running":
        return StepState("running" if _lock_alive(recording_session_job) else "interrupted", record)
    if record.get("state") in ("failed", "interrupted"):
        return StepState(record["state"], record)
    if record.get("inputs") != step_inputs(conversion_batch, recording_session_job, step, config_text):
        return StepState("changed", record)
    if record.get("code") != code and not ignore_code_changes:
        return StepState("code", record)
    return StepState(record.get("state", "pending"), record)


# --- locks ---------------------------------------------------------------------------------------------------------

def _lock_path(recording_session_job: RecordingSessionJob) -> Path:
    return runner_dir(recording_session_job) / "lock"


def _lock_owner(recording_session_job: RecordingSessionJob) -> dict[str, Any] | None:
    try:
        return json.loads(_lock_path(recording_session_job).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _lock_alive(recording_session_job: RecordingSessionJob) -> bool:
    owner = _lock_owner(recording_session_job)
    if owner is None:
        return False
    if owner.get("host") != socket.gethostname():
        return True  # another machine's runner: cannot check, take it as alive
    return psutil.pid_exists(int(owner.get("pid", -1)))  # not os.kill(pid, 0): on Windows it terminates


def acquire_lock(recording_session_job: RecordingSessionJob) -> bool:
    """Take the session's lock (False: another runner holds it). Stale locks of this host are taken over."""
    path = _lock_path(recording_session_job)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                        "started": datetime.now(UTC).isoformat(timespec="seconds")})
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_alive(recording_session_job):
                return False
            path.unlink(missing_ok=True)  # stale: its runner is gone
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(owner)
        return True
    return False


def _is_lock(relative: str) -> bool:
    """The session lock (not copied between a work dir and its local copy)."""
    return relative == "runner/lock"


def release_lock(recording_session_job: RecordingSessionJob) -> None:
    owner = _lock_owner(recording_session_job)
    if owner and owner.get("pid") == os.getpid() and owner.get("host") == socket.gethostname():
        _lock_path(recording_session_job).unlink(missing_ok=True)


# --- running ------------------------------------------------------------------------------------------------------

def pause_requested(conversion_batch: ConversionBatch) -> bool:
    return (conversion_batch.state_dir / PAUSE_FILE).exists()


def request_pause(conversion_batch: ConversionBatch) -> Path:
    conversion_batch.state_dir.mkdir(parents=True, exist_ok=True)
    path = conversion_batch.state_dir / PAUSE_FILE
    path.write_text(datetime.now(UTC).isoformat(timespec="seconds") + "\n", encoding="utf-8")
    return path


class Runner:
    def __init__(self, conversion_batch: ConversionBatch, steps: list[str] | None = None, ignore_code_changes: bool = False,
                 dry_run: bool = False, echo=None, source_cache: SourceCacheSettings | None = None):
        self.conversion_batch = conversion_batch
        self.source_cache_settings = source_cache or conversion_batch.source_cache  # source_cache: overrides the file
        self.source_cache = SourceCache(self.source_cache_settings, say=self._say, protected=[
            job.recording_session_path for job in conversion_batch.recording_session_jobs])
        self.steps = [s for s in conversion_batch.steps if steps is None or s in steps]
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

    def _skip_reason(self, recording_session_job: RecordingSessionJob, step: str) -> str | None:
        probes = recording_session_job.step_settings.get(step, {}).get("probes")
        if probes and not set(probes) & set(recording_session_job.recording_session_config.probes):
            return f"none of the probes {', '.join(probes)} in the session config"
        return None

    def _fresh(self, recording_session_job: RecordingSessionJob, step: str, previous: StepState, inputs: dict[str, Any],
               record: dict[str, Any] | None) -> None:
        """Remove what a step would reuse, unless it resumes an interrupted run with the same stamp."""
        if step != "spike-sort":
            return
        resume = (previous.state in ("interrupted", "failed") and record is not None
                  and record.get("inputs") == inputs and record.get("code") == self.code)
        if resume:
            self._say(f"{recording_session_job.recording_session_id} sort: resuming the interrupted sorting")
            return
        probes = (recording_session_job.step_settings.get("spike-sort", {}).get("probes")
                  or list(recording_session_job.recording_session_config.probes))
        scratch = sort_scratch_dir(self.conversion_batch, recording_session_job)
        for probe in probes:
            self._delete_tree(recording_session_job, recording_session_job.work_dir / "spike_sorting" / probe)
            for old in glob_in_work_folder(scratch, f"{recording_session_job.recording_session_path.name}_{probe}*"):
                self._delete_tree(recording_session_job, old)

    def _delete_tree(self, recording_session_job: RecordingSessionJob, folder: Path) -> None:
        """Delete ``folder`` only if it lies in the recording session's work dir or the sort scratch dir, and in no
        recording session folder (the source data are never touched)."""
        target = folder.resolve()
        allowed = [recording_session_job.work_dir.resolve(),
                   sort_scratch_dir(self.conversion_batch, recording_session_job).resolve()]
        protected = [s.recording_session_path.resolve() for s in self.conversion_batch.recording_session_jobs]
        if (not any(target != root and target.is_relative_to(root) for root in allowed)
                or any(target == source or target.is_relative_to(source) or source.is_relative_to(target)
                       for source in protected)):
            raise RuntimeError(f"refusing to delete {target}: not inside the work dir or the sort scratch dir of "
                               f"{recording_session_job.recording_session_id}, or it holds source data")
        remove_folder(target)

    def run_step(self, recording_session_job: RecordingSessionJob, step: str, config_path: Path, config_text: bytes,
                 staged: Staged | None = None, work_dir_origin: Path | None = None) -> bool:
        """Run ``step`` unless it is done (``work_dir_origin``: the work dir of ``recording_session_job`` is a local
        copy of that one; its records name the paths in it)."""
        previous = step_state(self.conversion_batch, recording_session_job, step, self.code, config_text, self.ignore_code_changes)
        if previous.finished:
            return True
        inputs = step_inputs(self.conversion_batch, recording_session_job, step, config_text)
        base = {"step": step, "inputs": inputs, "code": self.code, "run_id": uuid.uuid4().hex,
                "started": datetime.now(UTC).isoformat(timespec="seconds"), "host": socket.gethostname()}
        if reason := self._skip_reason(recording_session_job, step):
            if not self.dry_run:
                _write_record(recording_session_job, step, {**base, "state": "skipped", "reason": reason})
            self._say(f"{recording_session_job.recording_session_id} {step}: skipped ({reason})")
            return True
        source = staged.path if staged else recording_session_job.recording_session_path
        argv = [sys.executable, "-u", "-m", "ephysalis.cli.main", step, str(source),
                "--config", str(config_path), "--work-dir", str(recording_session_job.work_dir),
                *step_settings_argv(step, recording_session_job.step_settings[step])]
        if step in FORCE_STEPS:
            argv.append("--force")
        why = {"pending": "", "changed": " (inputs changed)", "code": " (code changed)", "failed": " (failed before)",
               "interrupted": " (interrupted before)"}.get(previous.state, "")
        if self.dry_run:
            self._say(f"{recording_session_job.recording_session_id} {step}: would run{why}")
            return True
        self._fresh(recording_session_job, step, previous, inputs, previous.record)
        log = runner_dir(recording_session_job) / "logs" / f"{step}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        record = {**base, "state": "running", "argv": argv[3:], "log": str(log), "pid": os.getpid()}
        if work_dir_origin is not None:
            local = str(recording_session_job.work_dir)
            record.update(argv=[str(work_dir_origin) + a[len(local):] if a == local or a.startswith(local + os.sep)
                                else a for a in argv[3:]],
                          log=str(work_dir_origin / log.relative_to(recording_session_job.work_dir)), worked_in=local)
        _write_record(recording_session_job, step, record)
        self._say(f"{recording_session_job.recording_session_id} {step}{why}")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as out:
            env = None
            if staged is not None and staged.copied:  # the step reads a copy: provenance records the original
                env = {**os.environ, ORIGIN_ENV: str(staged.origin)}
            if work_dir_origin is not None:  # the step works in a local copy: provenance records the work dir
                env = {**(env or os.environ), WORK_DIR_ORIGIN_ENV: str(work_dir_origin)}
            child = subprocess.Popen(argv, cwd=self.conversion_batch.root, stdout=out, stderr=subprocess.STDOUT, env=env)
            with self._children_lock:
                self._children.add(child)
            try:
                code = child.wait()
            finally:
                with self._children_lock:
                    self._children.discard(child)
        seconds = round(time.time() - t0)
        if self.stop.is_set() and code != 0:
            _write_record(recording_session_job, step, {**record, "state": "interrupted", "seconds": seconds})
            self._say(f"{recording_session_job.recording_session_id} {step}: interrupted")
            return False
        state = "done" if code == 0 else "failed"
        _write_record(recording_session_job, step, {**record, "state": state, "exit_code": code, "seconds": seconds,
                                      "finished": datetime.now(UTC).isoformat(timespec="seconds")})
        if code != 0:
            self._say(f"{recording_session_job.recording_session_id} {step}: failed (exit {code}), see {record['log']}")
        return code == 0

    def run_recording_session(self, recording_session_job: RecordingSessionJob) -> str:
        """``done``, ``failed``, ``paused``, ``interrupted`` or ``locked``."""
        if self.stop.is_set():
            return "interrupted"
        if not self.dry_run and not acquire_lock(recording_session_job):
            owner = _lock_owner(recording_session_job) or {}
            self._say(f"{recording_session_job.recording_session_id}: another runner works on it ({owner.get('host')}, pid {owner.get('pid')})")
            return "locked"
        try:
            if self.source_cache_settings.mode == "direct" or self.dry_run or not self._has_work(recording_session_job):
                return self._run_steps(recording_session_job)
            return self._run_steps_locally(recording_session_job)
        finally:
            if not self.dry_run:
                release_lock(recording_session_job)

    def _run_steps_locally(self, recording_session_job: RecordingSessionJob) -> str:
        """Run the steps in a local copy of the work dir, then copy it back (see the module docstring)."""
        name = recording_session_job.recording_session_id
        target = recording_session_job.work_dir
        local = replace(recording_session_job, work_dir=self.source_cache.work_dir(name))
        try:
            t0 = time.time()
            copied = mirror(target, local.work_dir, skip=_is_lock)
            if copied:
                self._say(f"{name}: work dir copied to {local.work_dir} ({copied / 1e9:.1f} GB in "
                          f"{(time.time() - t0) / 60:.1f} min)")
        except Exception as error:  # noqa: BLE001 (e.g. disk full): this session fails, the others go on
            self._say(f"{name}: copying the work dir to {local.work_dir} failed: {error}")
            return "failed"
        result = self._run_steps(local, work_dir_origin=target)
        try:
            t0 = time.time()
            copied = mirror(local.work_dir, target, skip=_is_lock)
            self._say(f"{name}: work dir copied back to {target} ({copied / 1e9:.1f} GB in "
                      f"{(time.time() - t0) / 60:.1f} min)")
        except Exception as error:  # noqa: BLE001: the session fails; its results stay in the local copy
            self._say(f"{name}: copying the work dir back to {target} failed: {error}; its results are in "
                      f"{local.work_dir} until the session runs again")
            return "failed"
        try:
            self.source_cache.remove_work_dir(name)
        except OSError as error:  # e.g. Windows: a file still open; replaced when the session runs again
            self._say(f"{name}: could not delete {local.work_dir} yet ({error})")
        return result

    def _run_steps(self, recording_session_job: RecordingSessionJob, work_dir_origin: Path | None = None) -> str:
        """``done``, ``failed``, ``paused`` or ``interrupted``: the steps of a locked session."""
        staged: Staged | None = None
        result = "failed"
        try:
            config_path = write_recording_session_config(recording_session_job) if not self.dry_run else recording_session_job.work_dir / "session.toml"
            with tempfile.TemporaryDirectory() as tmp:
                save_recording_session_config(recording_session_job.recording_session_config, Path(tmp) / "s.toml")
                config_text = (Path(tmp) / "s.toml").read_bytes()
            for step in self.steps:
                if self.stop.is_set():
                    result = "interrupted"
                    return result
                if pause_requested(self.conversion_batch):
                    result = "paused"
                    return result
                if staged is None and not self.dry_run and not step_state(
                        self.conversion_batch, recording_session_job, step, self.code, config_text,
                        self.ignore_code_changes).finished:  # the first step that runs: the folder to read
                    try:
                        staged = self.source_cache.stage(
                            recording_session_job.recording_session_id, recording_session_job.recording_session_path,
                            should_stop=lambda: self.stop.is_set() or pause_requested(self.conversion_batch))
                    except Exception as error:  # noqa: BLE001 (e.g. disk full): this session fails, the others go on
                        self._say(f"{recording_session_job.recording_session_id}: copying to the source cache failed: {error}")
                        return result
                    if staged is None:
                        result = "interrupted" if self.stop.is_set() else "paused"
                        return result
                if not self.run_step(recording_session_job, step, config_path, config_text, staged, work_dir_origin):
                    result = "interrupted" if self.stop.is_set() else "failed"
                    return result
            result = "done"
            return result
        finally:
            if staged is not None:
                self.source_cache.release(recording_session_job.recording_session_id, staged,
                                          keep=result in ("paused", "interrupted"))

    def _has_work(self, recording_session_job: RecordingSessionJob) -> bool:
        """Whether a step of the session has to run (so that its folder is read)."""
        with tempfile.TemporaryDirectory() as tmp:
            save_recording_session_config(recording_session_job.recording_session_config, Path(tmp) / "s.toml")
            config_text = (Path(tmp) / "s.toml").read_bytes()
        return any(not step_state(self.conversion_batch, recording_session_job, step, self.code, config_text,
                                  self.ignore_code_changes).finished for step in self.steps)

    def _prefetch(self, recording_session_jobs: list[RecordingSessionJob], ahead: int, finished: threading.Event) -> None:
        """Copy the sessions that have not started yet into the source cache, in order, while the running ones
        compute: at most ``ahead`` copies that no session reads yet, each only when it fits (it never takes the
        space of a running session)."""
        def should_stop() -> bool:
            return finished.is_set() or self.stop.is_set() or pause_requested(self.conversion_batch)

        for job in recording_session_jobs:
            while (not should_stop() and job.recording_session_id not in self._started
                   and len(self.source_cache.prefetched) >= ahead):
                finished.wait(5)
            if should_stop():
                return
            try:
                if job.recording_session_id in self._started or not self._has_work(job):
                    continue
                while not should_stop() and job.recording_session_id not in self._started:
                    if self.source_cache.prefetch(job.recording_session_id, job.recording_session_path, should_stop):
                        break
                    finished.wait(PREFETCH_RETRY_SECONDS)  # does not fit yet
            except Exception as error:  # noqa: BLE001 (a failed prefetch: the session copies when it starts)
                self._say(f"{job.recording_session_id}: prefetch failed: {error}")

    def _remove_left_work_dirs(self) -> None:
        """Delete the local work dirs a stopped runner left (a crash: its session's work dir has the state before)."""
        for job in self.conversion_batch.recording_session_jobs:
            if self.source_cache.work_dir(job.recording_session_id).exists() and not _lock_alive(job):
                self._say(f"{job.recording_session_id}: deleting the local work dir a stopped runner left")
                self.source_cache.remove_work_dir(job.recording_session_id)

    def run(self, recording_session_ids: list[str] | None = None, parallel: int = 1) -> dict[str, str]:
        recording_session_jobs = [recording_session_job for recording_session_job in self.conversion_batch.recording_session_jobs if recording_session_job.include
                            and (recording_session_ids is None or recording_session_job.recording_session_id in recording_session_ids)]
        if recording_session_ids:
            missing = set(recording_session_ids) - {recording_session_job.recording_session_id for recording_session_job in self.conversion_batch.recording_session_jobs}
            if missing:
                raise KeyError(f"not in {self.conversion_batch.conversion_batch_file}: {', '.join(sorted(missing))}")
        if not self.dry_run:
            (self.conversion_batch.state_dir / PAUSE_FILE).unlink(missing_ok=True)  # resume
        self._say(f"{self.conversion_batch.conversion_batch_name}: {len(recording_session_jobs)} sessions, {parallel} at a time, steps {', '.join(self.steps)}")
        if self.source_cache_settings.mode != "direct":
            cache = self.source_cache_settings
            self._say(f"source cache: {cache.mode} in {cache.dir}"
                      + (f", at most {cache.max_size_gb:g} GB" if cache.max_size_gb is not None else "")
                      + f", {cache.reserve_bytes / 1e9:g} GB left free")
        results: dict[str, str] = {}
        if self.source_cache_settings.mode != "direct" and not self.dry_run:
            self._remove_left_work_dirs()

        def run_one_recording_session(recording_session_job: RecordingSessionJob) -> None:
            self._started.add(recording_session_job.recording_session_id)
            t0 = time.time()
            result = self.run_recording_session(recording_session_job)
            results[recording_session_job.recording_session_id] = result
            if self.dry_run:
                return
            if result not in ("paused", "interrupted", "locked") or time.time() - t0 > 1:
                self._say(f"{recording_session_job.recording_session_id}: {result} ({(time.time() - t0) / 60:.0f} min)")
            if not self.dry_run:
                write_status(self.conversion_batch, [recording_session_job], self.code, self.ignore_code_changes)

        self._started: set[str] = set()
        finished = threading.Event()
        prefetcher = None
        if self.source_cache_settings.mode != "direct" and not self.dry_run:
            prefetcher = threading.Thread(target=self._prefetch, args=(recording_session_jobs, max(1, parallel), finished),
                                          name="prefetch", daemon=True)
            prefetcher.start()
        try:
            with ThreadPoolExecutor(max(1, parallel)) as pool:
                list(pool.map(run_one_recording_session, recording_session_jobs))
        finally:
            finished.set()
            if prefetcher is not None:
                prefetcher.join()
                self.source_cache.keep_prefetched()
        if self.dry_run:
            self._say("dry run: nothing was run")
            return results
        result_counts = {result: list(results.values()).count(result) for result in sorted(set(results.values()))}
        self._say(", ".join(f"{count} {result}" for result, count in result_counts.items()) or "no sessions")
        if pause_requested(self.conversion_batch):
            self._say(f"paused; resume with: ephysalis run {self.conversion_batch.conversion_batch_file}")
        return results


# --- status -------------------------------------------------------------------------------------------------------

SYMBOLS = {"done": "+", "skipped": "-", "pending": ".", "changed": "~", "code": "c", "failed": "X",
           "interrupted": "!", "running": ">"}


def recording_session_states(conversion_batch: ConversionBatch, recording_session_job: RecordingSessionJob, code: dict[str, Any],
                   ignore_code_changes: bool = False) -> dict[str, StepState]:
    with tempfile.TemporaryDirectory() as tmp:
        save_recording_session_config(recording_session_job.recording_session_config, Path(tmp) / "s.toml")
        config_text = (Path(tmp) / "s.toml").read_bytes()
    states: dict[str, StepState] = {}
    for step in conversion_batch.steps:
        state = step_state(conversion_batch, recording_session_job, step, code, config_text, ignore_code_changes)
        if state.finished and any(dep in states and not states[dep].finished for dep in STEP_INPUTS[step]):
            state = StepState("changed", state.record)  # runs again after the step before it
        states[step] = state
    return states


def status_table(conversion_batch: ConversionBatch, code: dict[str, Any] | None = None, ignore_code_changes: bool = False,
                 recording_session_ids: list[str] | None = None) -> str:
    code = code if code is not None else code_versions()
    width = max([len(recording_session_job.recording_session_id) for recording_session_job in conversion_batch.recording_session_jobs] + [7])
    step_columns = [step.replace("extract-", "x-").replace("align-clocks", "clocks") for step in conversion_batch.steps]
    lines = [f"{'recording_session':{width}}  " + " ".join(step_columns)]
    for recording_session_job in conversion_batch.recording_session_jobs:
        if recording_session_ids and recording_session_job.recording_session_id not in recording_session_ids:
            continue
        if not recording_session_job.include:
            lines.append(f"{recording_session_job.recording_session_id:{width}}  (excluded)")
            continue
        states = recording_session_states(conversion_batch, recording_session_job, code, ignore_code_changes)
        cells = [f"{SYMBOLS[states[step].state]:^{len(column)}}" for step, column in zip(conversion_batch.steps, step_columns)]
        lines.append(f"{recording_session_job.recording_session_id:{width}}  " + " ".join(cells))
    lines.append("+ done  - skipped  . not run  ~ inputs changed  c code changed  X failed  ! interrupted  > running")
    if pause_requested(conversion_batch):
        lines.append(f"paused ({conversion_batch.state_dir / PAUSE_FILE})")
    return "\n".join(lines)


def write_status(conversion_batch: ConversionBatch, recording_session_jobs: list[RecordingSessionJob], code: dict[str, Any],
                 ignore_code_changes: bool = False) -> None:
    """Update ``state_dir/status.tsv`` (session x step states) for ``recording_session_jobs``."""
    path = conversion_batch.state_dir / "status.tsv"
    with _STATUS_LOCK:
        rows: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines()[1:]:
                rows[line.split("\t", 1)[0]] = line
        for recording_session_job in recording_session_jobs:
            states = recording_session_states(conversion_batch, recording_session_job, code, ignore_code_changes)
            rows[recording_session_job.recording_session_id] = "\t".join([recording_session_job.recording_session_id,
                                                          *(states[step].state for step in conversion_batch.steps),
                                                          datetime.now().isoformat(timespec="seconds")])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\t".join(["recording_session", *conversion_batch.steps, "updated"]) + "\n"
                        + "\n".join(rows[k] for k in sorted(rows)) + "\n", encoding="utf-8")


def install_signal_handlers(runner: Runner) -> None:
    def handler(signum, frame):
        runner._say("stopping: terminating the running steps (they run again on resume)")
        runner.interrupt()

    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)
