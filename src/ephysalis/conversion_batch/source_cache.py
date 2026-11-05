"""How the steps of ``ephysalis run`` read the recording session folders: ``[source_cache]`` of the batch.

Modes (:class:`SourceCacheMode`):

- ``direct``: the steps read the folders where they are (default).
- ``memory`` (Linux only): before the first step of a session that has to run, the runner copies its folder to
  the shared memory (``dir``, default ``/dev/shm/ephysalis``) and the steps read the copy; it is deleted when the
  session ends.
- ``local``: the same, into ``dir`` on a local disk (required). A session that is paused or interrupted keeps its
  copy: it is reused when the session resumes (if the source files are unchanged) and deleted when its space is
  needed.

The copies of all sessions together (of every runner on this machine) stay below ``max_size_gb``, and leave
``reserve_gb`` free: of the memory (memory: what the kernel counts as available) or of the disk (local). A session
that needs more space than is free waits until other sessions release theirs; a session that alone is bigger than
the cache can ever hold is read directly. Reading a folder once, sequentially, is fast on a network file system;
the steps' own reads (memory-mapped, several workers) fetch the same data several times.

Layout of ``dir``: ``<recording_session_id>/`` (the copy), ``<recording_session_id>.json`` (its entry: source, size,
files, state, owner) and ``.lock`` (taken while the entries are changed).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import psutil
from filelock import FileLock

logger = logging.getLogger(__name__)

GB = 1_000_000_000
DEFAULT_MEMORY_DIR = "/dev/shm/ephysalis"
DEFAULT_RESERVE_GB = {"memory": 32.0, "local": 10.0}
WAIT_SECONDS = 30.0
ORIGIN_ENV = "EPHYSALIS_RECORDING_SESSION_ORIGIN"  # the original folder of a step that reads a copy (provenance)


class SourceCacheMode(StrEnum):
    DIRECT = "direct"
    MEMORY = "memory"
    LOCAL = "local"


@dataclass(frozen=True)
class SourceCacheSettings:
    mode: SourceCacheMode = SourceCacheMode.DIRECT
    dir: Path | None = None  # memory: default DEFAULT_MEMORY_DIR; local: required
    max_size_gb: float | None = None  # all copies together (None: only the free memory or disk space limits them)
    reserve_gb: float | None = None  # left free (default DEFAULT_RESERVE_GB of the mode)

    @property
    def reserve_bytes(self) -> int:
        reserve = self.reserve_gb if self.reserve_gb is not None else DEFAULT_RESERVE_GB.get(self.mode, 0.0)
        return int(reserve * GB)


def source_cache_settings(data: dict[str, Any], root: Path, source: object) -> SourceCacheSettings:
    """``[source_cache]`` of a batch file (``dir`` relative to ``root``)."""
    if unknown := set(data) - {"mode", "dir", "max_size_gb", "reserve_gb"}:
        raise ValueError(f"{source}: unknown keys of [source_cache]: {', '.join(sorted(unknown))} "
                         f"(known: mode, dir, max_size_gb, reserve_gb)")
    try:
        mode = SourceCacheMode(data.get("mode", "direct"))
    except ValueError:
        raise ValueError(f"{source}: [source_cache] mode {data['mode']!r} is not one of "
                         f"{', '.join(m.value for m in SourceCacheMode)}") from None
    folder = Path(data["dir"]) if data.get("dir") else None
    if folder is not None and not folder.is_absolute():
        folder = root / folder
    for key in ("max_size_gb", "reserve_gb"):
        value = data.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float) or value < 0
                                  or (key == "max_size_gb" and value == 0)):
            raise ValueError(f"{source}: [source_cache] {key} must be a positive number of GB, not {value!r}")
    return check_source_cache_settings(SourceCacheSettings(
        mode=mode, dir=folder.resolve() if folder else None,
        max_size_gb=float(data["max_size_gb"]) if data.get("max_size_gb") is not None else None,
        reserve_gb=float(data["reserve_gb"]) if data.get("reserve_gb") is not None else None), source)


def check_source_cache_settings(settings: SourceCacheSettings, source: object) -> SourceCacheSettings:
    """``settings`` with the default dir of memory mode; local mode needs a dir. (That memory mode runs only on
    Linux is checked when a runner starts: the file stays readable, e.g. for ``ephysalis status``, everywhere.)"""
    if settings.mode == SourceCacheMode.LOCAL and settings.dir is None:
        raise ValueError(f"{source}: [source_cache] mode = \"local\" needs dir (a folder on a local disk)")
    if settings.mode == SourceCacheMode.MEMORY and settings.dir is None:
        return SourceCacheSettings(settings.mode, Path(DEFAULT_MEMORY_DIR), settings.max_size_gb, settings.reserve_gb)
    return settings


def folder_files(folder: Path) -> dict[str, list[int]]:
    """``{relative path: [size, mtime_ns]}`` of the files below ``folder`` (symbolic links followed)."""
    files = {}
    for parent, _, names in os.walk(folder, followlinks=True):
        for name in names:
            path = Path(parent) / name
            stat = path.stat()
            files[path.relative_to(folder).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    return dict(sorted(files.items()))


@dataclass
class Staged:
    """The folder the steps of a session read, and where it came from."""
    path: Path
    origin: Path
    entry: Path | None = None  # the cache entry (None: read directly)

    @property
    def copied(self) -> bool:
        return self.entry is not None


class SourceCache:
    """Copies of recording session folders (:mod:`ephysalis.conversion_batch.source_cache`)."""

    def __init__(self, settings: SourceCacheSettings, protected: list[Path] | None = None,
                 say: Callable[[str], None] | None = None):
        self.settings = settings
        self.protected = [p.resolve() for p in protected or []]  # never deleted: the source folders
        self.say = say or (lambda text: logger.info(text))
        self.owner = {"host": socket.gethostname(), "pid": os.getpid()}
        if settings.mode == SourceCacheMode.MEMORY and not sys.platform.startswith("linux"):
            raise ValueError(f"[source_cache] mode = \"memory\" needs Linux (/dev/shm); on {sys.platform} use "
                             f"mode = \"local\" (ephysalis run --source-cache local --source-cache-dir ...)")
        if settings.mode != SourceCacheMode.DIRECT:
            assert settings.dir is not None
            folder = settings.dir.resolve()
            for source in self.protected:
                if folder == source or folder.is_relative_to(source) or source.is_relative_to(folder):
                    raise ValueError(f"[source_cache] dir {folder} overlaps the recording session folder {source}")

    @property
    def dir(self) -> Path:
        assert self.settings.dir is not None
        return self.settings.dir

    # --- the entries --------------------------------------------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self.dir / ".lock"):
            yield

    def _entries(self) -> dict[str, dict[str, Any]]:
        entries = {}
        for path in self.dir.glob("*.json"):
            try:
                entries[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                entries[path.stem] = {"size": 0, "state": "broken"}
        return entries

    def _write_entry(self, name: str, entry: dict[str, Any]) -> None:
        path = self.dir / f"{name}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, indent=1), encoding="utf-8")
        tmp.replace(path)

    def _in_use(self, entry: dict[str, Any]) -> bool:
        owner = entry.get("owner")
        if not owner:
            return False
        if owner.get("host") != self.owner["host"]:
            return True  # another machine's runner: cannot check
        return psutil.pid_exists(int(owner.get("pid", -1)))

    def _remove(self, name: str) -> None:
        """Delete the copy ``name`` and its entry (only inside the cache dir, never a source folder)."""
        for target in (self.dir / name, self.dir / f"{name}.partial"):
            resolved = target.resolve() if target.exists() else target
            if not resolved.is_relative_to(self.dir.resolve()) or resolved == self.dir.resolve() or any(
                    resolved == p or resolved.is_relative_to(p) or p.is_relative_to(resolved) for p in self.protected):
                raise RuntimeError(f"refusing to delete {resolved}: not a copy in the source cache {self.dir}")
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
        (self.dir / f"{name}.json").unlink(missing_ok=True)

    def _not_yet_written(self, entries: dict[str, dict[str, Any]]) -> int:
        """Bytes that copies being made have reserved but not yet written (the free space does not show them)."""
        pending = 0
        for name, entry in entries.items():
            if entry.get("state") == "copying" and self._in_use(entry):
                partial = self.dir / f"{name}.partial"
                written = sum(p.stat().st_size for p in partial.rglob("*") if p.is_file()) if partial.is_dir() else 0
                pending += max(0, int(entry.get("size", 0)) - written)
        return pending

    def _limits(self, used: int, pending: int) -> tuple[int, str]:
        """Bytes a new copy may take now, and what limits it."""
        limits = []
        if self.settings.max_size_gb is not None:
            limits.append((int(self.settings.max_size_gb * GB) - used, f"max_size_gb {self.settings.max_size_gb:g}"))
        free = shutil.disk_usage(self.dir).free - self.settings.reserve_bytes - pending
        limits.append((free, f"free space of {self.dir}"))
        if self.settings.mode == SourceCacheMode.MEMORY:
            limits.append((psutil.virtual_memory().available - self.settings.reserve_bytes - pending, "available memory"))
        return min(limits)

    def _capacity(self) -> int | None:
        """The most the cache can ever hold (None: no fixed bound)."""
        if self.settings.max_size_gb is None:
            return None
        return int(self.settings.max_size_gb * GB)

    # --- staging ------------------------------------------------------------------------------------------------

    def stage(self, name: str, source: Path, should_stop: Callable[[], bool] | None = None) -> Staged | None:
        """The folder the steps of session ``name`` read: ``source`` (direct, or too big) or its copy, after
        waiting for space. None: ``should_stop`` became true while waiting."""
        source = source.resolve()
        if self.settings.mode == SourceCacheMode.DIRECT:
            return Staged(source, source)
        files = folder_files(source)
        size = sum(f[0] for f in files.values())
        capacity = self._capacity()
        if capacity is not None and size > capacity:
            self.say(f"{name}: {size / GB:.1f} GB is more than the source cache holds "
                     f"(max_size_gb {self.settings.max_size_gb:g}); read directly")
            return Staged(source, source)
        waited = False
        while True:
            with self._locked():
                entries = self._entries()
                mine = entries.get(name)
                if (mine and mine.get("state") == "complete" and mine.get("source") == str(source)
                        and mine.get("files") == files and (self.dir / name).is_dir() and not self._in_use(mine)):
                    self._write_entry(name, {**mine, "owner": self.owner})
                    self.say(f"{name}: reading the copy in {self.dir} (kept from before)")
                    return Staged(self.dir / name, source, self.dir / f"{name}.json")
                if mine and self._in_use(mine):
                    raise RuntimeError(f"{name}: its copy in {self.dir} is used by another runner ({mine['owner']})")
                if mine:
                    self._remove(name)  # outdated or broken
                    entries.pop(name)
                # stale copies of other sessions (their runner is gone) count until they are deleted, which
                # happens when their space is needed
                stale = [n for n, e in entries.items() if not self._in_use(e)]
                used = sum(int(e.get("size", 0)) for e in entries.values())
                free, limit = self._limits(used, self._not_yet_written(entries))
                stale_size = sum(int(entries[n].get("size", 0)) for n in stale)
                if size <= free + stale_size:
                    for n in stale:
                        if size <= free:
                            break
                        self._remove(n)
                        free += int(entries[n].get("size", 0))
                    if size <= free:
                        self._write_entry(name, {"source": str(source), "size": size, "files": files,
                                                 "state": "copying", "owner": self.owner})
                        break
                if not any(self._in_use(e) for e in entries.values()):  # waiting would not free anything
                    self.say(f"{name}: {size / GB:.1f} GB do not fit ({limit}: {free / GB:.1f} GB) and no other "
                             f"session holds space; read directly")
                    return Staged(source, source)
            if not waited:
                self.say(f"{name}: waiting for {size / GB:.1f} GB in the source cache ({limit}: {free / GB:.1f} GB free)")
                waited = True
            if should_stop and should_stop():
                return None
            time.sleep(WAIT_SECONDS)
        return self._copy(name, source, files, size)

    def _copy(self, name: str, source: Path, files: dict[str, list[int]], size: int) -> Staged:
        partial, target = self.dir / f"{name}.partial", self.dir / name
        if partial.exists():
            shutil.rmtree(partial)
        t0 = time.time()
        self.say(f"{name}: copying {size / GB:.1f} GB to {self.dir}")
        try:
            for relative, (file_size, mtime_ns) in files.items():
                destination = partial / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, destination)
                if destination.stat().st_size != file_size:
                    raise OSError(f"{source / relative}: copied {destination.stat().st_size} of {file_size} bytes")
                os.utime(destination, ns=(mtime_ns, mtime_ns))
            partial.rename(target)
        except BaseException:
            with self._locked():
                self._remove(name)
            raise
        with self._locked():
            entry = json.loads((self.dir / f"{name}.json").read_text(encoding="utf-8"))
            self._write_entry(name, {**entry, "state": "complete"})
        seconds = max(time.time() - t0, 1e-3)
        self.say(f"{name}: copied in {seconds / 60:.1f} min ({size / seconds / 1e6:.0f} MB/s)")
        return Staged(target, source, self.dir / f"{name}.json")

    def release(self, name: str, staged: Staged, keep: bool = False) -> None:
        """The session is over: delete its copy, or (``keep``, local mode) leave it for a resume."""
        if not staged.copied:
            return
        with self._locked():
            entry = self._entries().get(name)
            if entry is None or entry.get("owner") != self.owner:
                return
            if keep and self.settings.mode == SourceCacheMode.LOCAL and entry.get("state") == "complete":
                self._write_entry(name, {**entry, "owner": None})
                return
            try:
                self._remove(name)
            except OSError as error:  # e.g. Windows: a file still open; deleted later, when its space is needed
                self._write_entry(name, {**entry, "owner": None})
                self.say(f"{name}: could not delete its copy in {self.dir} yet ({error}); it is deleted when its "
                         f"space is needed")
