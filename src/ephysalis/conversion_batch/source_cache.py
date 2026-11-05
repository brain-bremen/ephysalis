"""How the steps of ``ephysalis run`` read the recording session folders: ``[source_cache]`` of the batch.

Modes (:class:`SourceCacheMode`):

- ``direct``: the steps read the folders where they are (default).
- ``local``: before the first step of a session that has to run, the runner copies its folder into ``dir`` on a
  local disk and the steps read the copy; it is deleted when the session ends. While sessions compute, the runner
  prefetches the next ones (:meth:`SourceCache.prefetch`): at most ``parallel`` copies ahead, each only when it
  fits; for a running session that needs space, prefetched copies make room first. A session that is paused or
  interrupted keeps its copy: it is reused when the session resumes (if the source files are unchanged) and deleted
  when its space is needed.

  The work dirs stay local too: when a step of a session has to run, the runner copies the session's work dir into
  ``dir/.work/<recording_session_id>`` (:meth:`SourceCache.work_dir`), the steps read and write that copy, and when
  the session ends (done, failed, paused or interrupted) it is copied back (:func:`mirror`) and deleted. So the
  steps' outputs (LFP, CSD, MUA, ...) are written to the work dir once, not read back by the later steps over the
  network, and the export builds its files locally and copies each into the dataset in one go. Until a session ends,
  ``ephysalis status`` shows the state its work dir had when the session started.

The copies of all sessions together (of every runner on this machine), with the local work dirs as they are when
a copy is made, stay below ``max_size_gb``, and leave ``reserve_gb`` of the disk free (the work dirs grow while the
steps run, by the size of a session's outputs). A session
that needs more space than is free waits until other sessions release theirs; a session that alone is bigger than
the cache can ever hold is read directly. Reading a folder once, sequentially, is fast on a network file system;
the steps' own reads (memory-mapped, several workers) fetch the same data several times.

Layout of ``dir``: ``<recording_session_id>/`` (the copy), ``<recording_session_id>.json`` (its entry: source, size,
files, state, owner), ``.work/<recording_session_id>/`` (a local work dir) and ``.lock`` (taken while the entries
are changed).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import threading
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
DEFAULT_RESERVE_GB = 10.0
WAIT_SECONDS = 30.0
ORIGIN_ENV = "EPHYSALIS_RECORDING_SESSION_ORIGIN"  # the original folder of a step that reads a copy (provenance)
WORK_DIR_ORIGIN_ENV = "EPHYSALIS_WORK_DIR_ORIGIN"  # the work dir a step's local work dir stands for (provenance)
WORK_DIRS = ".work"  # the local work dirs, in the cache dir
MIRROR_LAST = "runner/"  # the runner's step records: copied last, so that a record never claims outputs not copied


class SourceCacheMode(StrEnum):
    DIRECT = "direct"
    LOCAL = "local"


@dataclass(frozen=True)
class SourceCacheSettings:
    mode: SourceCacheMode = SourceCacheMode.DIRECT
    dir: Path | None = None  # local: required
    max_size_gb: float | None = None  # all copies together (None: only the free disk space limits them)
    reserve_gb: float | None = None  # left free on the disk (default DEFAULT_RESERVE_GB)

    @property
    def reserve_bytes(self) -> int:
        return int((self.reserve_gb if self.reserve_gb is not None else DEFAULT_RESERVE_GB) * GB)


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
    """``settings``, checked: local mode needs a dir."""
    if settings.mode == SourceCacheMode.LOCAL and settings.dir is None:
        raise ValueError(f"{source}: [source_cache] mode = \"local\" needs dir (a folder on a local disk)")
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


def mirror(source: Path, target: Path, skip: Callable[[str], bool] | None = None) -> int:
    """Make ``target`` a copy of the folder ``source``: copy the files that differ from it (size or modification
    time), delete the ones ``source`` does not have (and the folders left empty); the relative paths ``skip`` names
    are left alone in both. Returns the bytes copied.

    The step records (``runner/``) that differ are deleted first and copied last: when the copy stops half-way, the
    steps whose outputs changed have no record in ``target`` (they run again), and the others keep theirs."""
    skip = skip or (lambda relative: False)
    have = {r: f for r, f in folder_files(source).items() if not skip(r)}
    old = {r: f for r, f in (folder_files(target) if target.is_dir() else {}).items() if not skip(r)}
    records = [r for r in sorted(set(have) | set(old)) if r.startswith(MIRROR_LAST)]
    for relative in records:
        if old.get(relative) != have.get(relative) and relative in old:
            (target / relative).unlink()
    copied = 0
    for relative in [r for r in have if not r.startswith(MIRROR_LAST)] + [r for r in records if r in have]:
        size, mtime_ns = have[relative]
        if old.get(relative) == have[relative]:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".mirror-partial")
        shutil.copyfile(source / relative, tmp)
        if tmp.stat().st_size != size:
            raise OSError(f"{source / relative}: copied {tmp.stat().st_size} of {size} bytes")
        os.utime(tmp, ns=(mtime_ns, mtime_ns))
        tmp.replace(destination)
        copied += size
    for relative in old:
        if relative not in have and not relative.startswith(MIRROR_LAST) and (target / relative).exists():
            (target / relative).unlink()
    for parent, folders, files in os.walk(target, topdown=False):
        folder = Path(parent)
        if folder != target and not folders and not files and not (source / folder.relative_to(target)).is_dir():
            folder.rmdir()
    return copied


def folder_bytes(folder: Path) -> int:
    """Bytes of the files below ``folder`` (0: no such folder), while steps may write there (files that are renamed
    or deleted meanwhile do not count)."""
    total = 0
    for parent, _, names in os.walk(folder):
        for name in names:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


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
    """Copies of recording session folders (:mod:`ephysalis.conversion_batch.source_cache`). One object per
    runner; its threads (the sessions and the prefetch) share it."""

    def __init__(self, settings: SourceCacheSettings, protected: list[Path] | None = None,
                 say: Callable[[str], None] | None = None):
        self.settings = settings
        self.protected = [p.resolve() for p in protected or []]  # never deleted: the source folders
        self.say = say or (lambda text: logger.info(text))
        self.owner = {"host": socket.gethostname(), "pid": os.getpid()}
        self._copying: dict[str, threading.Event] = {}  # copies this runner is making (set when done)
        self._prefetching: set[str] = set()  # copies being made ahead
        self._prefetched: set[str] = set()  # copies made ahead that no session reads yet
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

    def work_dir(self, name: str) -> Path:
        """The local work dir of session ``name``."""
        return self.dir / WORK_DIRS / name

    def remove_work_dir(self, name: str) -> None:
        """Delete the local work dir of session ``name`` (only inside the cache dir, never a source folder)."""
        target = self.work_dir(name)
        if target.is_symlink() or not target.exists():
            target.unlink(missing_ok=True)
            return
        resolved = target.resolve()
        if not resolved.is_relative_to((self.dir / WORK_DIRS).resolve()) or any(
                resolved == p or resolved.is_relative_to(p) or p.is_relative_to(resolved) for p in self.protected):
            raise RuntimeError(f"refusing to delete {resolved}: not a work dir in the source cache {self.dir}")
        shutil.rmtree(target)

    @property
    def prefetched(self) -> set[str]:
        """Sessions copied (or being copied) ahead that no session reads yet."""
        return self._prefetched | self._prefetching

    # --- the entries --------------------------------------------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self.dir / ".lock"):  # also between the threads of this runner
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
        self._prefetched.discard(name)

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
        return min(limits)

    def _capacity(self) -> int | None:
        """The most the cache can ever hold (None: no fixed bound)."""
        if self.settings.max_size_gb is None:
            return None
        return int(self.settings.max_size_gb * GB)

    def _reusable(self, entry: dict[str, Any] | None, name: str, source: Path, files: dict[str, list[int]]) -> bool:
        return bool(entry and entry.get("state") == "complete" and entry.get("source") == str(source)
                    and entry.get("files") == files and (self.dir / name).is_dir()
                    and (not self._in_use(entry) or entry.get("owner") == self.owner))

    # --- staging ------------------------------------------------------------------------------------------------

    def stage(self, name: str, source: Path, should_stop: Callable[[], bool] | None = None) -> Staged | None:
        """The folder the steps of session ``name`` read: ``source`` (direct, or too big) or its copy (made now,
        prefetched or kept from before), after waiting for space. None: ``should_stop`` became true meanwhile."""
        source = source.resolve()
        if self.settings.mode == SourceCacheMode.DIRECT:
            return Staged(source, source)
        result = self._stage(name, source, should_stop, prefetch=False)
        return result if isinstance(result, Staged) or result is None else None

    def prefetch(self, name: str, source: Path, should_stop: Callable[[], bool] | None = None) -> bool:
        """Copy session ``name`` ahead if it fits now (without waiting; only copies of gone runners make room).
        True: its copy is there (now or before); False: it does not fit now (or is read directly)."""
        if self.settings.mode == SourceCacheMode.DIRECT:
            return False
        return self._stage(name, source.resolve(), should_stop, prefetch=True) is True

    def _stage(self, name: str, source: Path, should_stop: Callable[[], bool] | None, prefetch: bool):
        files = folder_files(source)
        size = sum(f[0] for f in files.values())
        capacity = self._capacity()
        if capacity is not None and size > capacity:
            if prefetch:
                return False
            self.say(f"{name}: {size / GB:.1f} GB is more than the source cache holds "
                     f"(max_size_gb {self.settings.max_size_gb:g}); read directly")
            return Staged(source, source)
        waited = False
        while True:
            pending_copy: threading.Event | None = None
            with self._locked():
                entries = self._entries()
                mine = entries.get(name)
                if name in self._copying:  # this runner copies it already (the prefetch)
                    if prefetch:
                        return True
                    pending_copy = self._copying[name]
                elif self._reusable(mine, name, source, files):
                    if prefetch:
                        return True
                    how = "prefetched" if name in self._prefetched else "kept from before"
                    self._prefetched.discard(name)
                    self._prefetching.discard(name)
                    self._write_entry(name, {**mine, "owner": self.owner})
                    self.say(f"{name}: reading the copy in {self.dir} ({how})")
                    return Staged(self.dir / name, source, self.dir / f"{name}.json")
                else:
                    if mine and self._in_use(mine) and mine.get("owner") != self.owner:
                        raise RuntimeError(f"{name}: its copy in {self.dir} is used by another runner ({mine['owner']})")
                    if mine:
                        self._remove(name)  # outdated or broken
                        entries.pop(name)
                    # room is made from the copies of gone runners and (not for a prefetch) the copies this runner
                    # prefetched for later sessions; until then they count
                    stale = [n for n, e in entries.items() if not self._in_use(e)]
                    if not prefetch:
                        stale += sorted(n for n in self._prefetched if n in entries and n not in self._copying)
                    used = sum(int(e.get("size", 0)) for e in entries.values()) + folder_bytes(self.dir / WORK_DIRS)
                    free, limit = self._limits(used, self._not_yet_written(entries))
                    if size <= free + sum(int(entries[n].get("size", 0)) for n in stale):
                        for n in reversed(stale):  # the prefetched copies of the latest sessions first
                            if size <= free:
                                break
                            if n in self._prefetched:
                                self.say(f"{name}: deleting the prefetched copy of {n} to make room")
                            self._remove(n)
                            free += int(entries[n].get("size", 0))
                        self._write_entry(name, {"source": str(source), "size": size, "files": files,
                                                 "state": "copying", "owner": self.owner})
                        self._copying[name] = threading.Event()
                        if prefetch:
                            self._prefetching.add(name)
                        break
                    if prefetch:
                        return False
                    holders = [n for n, e in entries.items() if self._in_use(e) and n not in stale]
                    if not holders:  # waiting would not free anything
                        self.say(f"{name}: {size / GB:.1f} GB do not fit ({limit}: {free / GB:.1f} GB) and no other "
                                 f"session holds space; read directly")
                        return Staged(source, source)
            if pending_copy is not None:  # the prefetch of this session: wait for its copy, then take it
                while not pending_copy.wait(1.0):
                    if should_stop and should_stop():
                        return None
                continue
            if not waited:
                self.say(f"{name}: waiting for {size / GB:.1f} GB in the source cache ({limit}: {free / GB:.1f} GB free)")
                waited = True
            if should_stop and should_stop():
                return None
            time.sleep(WAIT_SECONDS)
        staged = self._copy(name, source, files, size, should_stop, prefetch)
        if prefetch:
            return staged is not None
        return staged

    def _copy(self, name: str, source: Path, files: dict[str, list[int]], size: int,
              should_stop: Callable[[], bool] | None, prefetch: bool) -> Staged | None:
        partial, target = self.dir / f"{name}.partial", self.dir / name
        t0 = time.time()
        self.say(f"{name}: copying {size / GB:.1f} GB to {self.dir}" + (" (prefetch)" if prefetch else ""))
        try:  # whatever fails: the partial copy and the entry go (they are never read)
            if partial.exists():
                shutil.rmtree(partial)
            for relative, (file_size, mtime_ns) in files.items():
                if should_stop and should_stop():
                    raise _Stopped
                destination = partial / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, destination)
                if destination.stat().st_size != file_size:
                    raise OSError(f"{source / relative}: copied {destination.stat().st_size} of {file_size} bytes")
                os.utime(destination, ns=(mtime_ns, mtime_ns))
            partial.rename(target)
        except BaseException as error:
            with self._locked():
                self._remove(name)
                self._prefetching.discard(name)
                self._copying.pop(name).set()
            if isinstance(error, _Stopped):
                return None
            raise
        with self._locked():
            entry = json.loads((self.dir / f"{name}.json").read_text(encoding="utf-8"))
            self._write_entry(name, {**entry, "state": "complete"})
            if prefetch and name in self._prefetching:
                self._prefetched.add(name)
            self._prefetching.discard(name)
            self._copying.pop(name).set()
        seconds = max(time.time() - t0, 1e-3)
        self.say(f"{name}: copied in {seconds / 60:.1f} min ({size / seconds / 1e6:.0f} MB/s)")
        return Staged(target, source, self.dir / f"{name}.json")

    def release(self, name: str, staged: Staged, keep: bool = False) -> None:
        """The session is over: delete its copy, or (``keep``) leave it for a resume."""
        if not staged.copied:
            return
        with self._locked():
            entry = self._entries().get(name)
            if entry is None or entry.get("owner") != self.owner:
                return
            if keep and entry.get("state") == "complete":
                self._write_entry(name, {**entry, "owner": None})
                return
            try:
                self._remove(name)
            except OSError as error:  # e.g. Windows: a file still open; deleted later, when its space is needed
                self._write_entry(name, {**entry, "owner": None})
                self.say(f"{name}: could not delete its copy in {self.dir} yet ({error}); it is deleted when its "
                         f"space is needed")

    def keep_prefetched(self) -> None:
        """The runner ends: the prefetched copies no session read are kept for the next run (no owner)."""
        with self._locked():
            entries = self._entries()
            for name in sorted(self._prefetched):
                if name in entries and entries[name].get("owner") == self.owner:
                    self._write_entry(name, {**entries[name], "owner": None})
            self._prefetched.clear()


class _Stopped(Exception):
    """A copy was stopped (the runner pauses or stops)."""
