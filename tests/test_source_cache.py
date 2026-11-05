"""[source_cache] of a conversion batch: the steps read the session folders directly, or a copy in memory or on a
local disk, below a size limit (ephysalis.conversion_batch.source_cache)."""

import json
import threading
import time
from pathlib import Path

import pytest
from test_conversion_batch import STEPS, _states, write_conversion_batch

from ephysalis.cli.main import main
from ephysalis.conversion_batch import source_cache as sc
from ephysalis.conversion_batch.config import load_conversion_batch
from ephysalis.conversion_batch.runner import Runner, read_record
from ephysalis.conversion_batch.source_cache import (
    GB,
    SourceCache,
    SourceCacheMode,
    SourceCacheSettings,
    folder_files,
)


def _session(root: Path, name: str, sizes: dict[str, int]) -> Path:
    folder = root / "sources" / name
    for relative, size in sizes.items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_bytes(bytes(range(256)) * (size // 256) + b"x" * (size % 256))
    return folder


def _cache(tmp_path: Path, mode="local", max_size_gb=None, protected=(), lines=None) -> SourceCache:
    settings = SourceCacheSettings(SourceCacheMode(mode), tmp_path / "cache", max_size_gb, reserve_gb=0)
    return SourceCache(settings, protected=list(protected), say=(lines.append if lines is not None else None))


def test_settings_of_the_batch_file(tmp_path):
    path = write_conversion_batch(tmp_path)
    assert load_conversion_batch(path).source_cache.mode == "direct"  # default
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "memory"\n')
    cache = load_conversion_batch(path).source_cache
    assert cache.mode == "memory" and cache.dir == Path(sc.DEFAULT_MEMORY_DIR) and cache.reserve_bytes == 32 * GB
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "local"\ndir = "cache"\nmax_size_gb = 500\n')
    cache = load_conversion_batch(path).source_cache
    assert cache.dir == (tmp_path / "cache").resolve() and cache.max_size_gb == 500.0 and cache.reserve_bytes == 10 * GB
    for extra, message in [('mode = "ram"', "is not one of direct, memory, local"),
                           ('mode = "local"', "needs dir"),
                           ('mode = "local"\ndir = "c"\nmax_size_gb = 0', "positive number"),
                           ('mode = "local"\ndir = "c"\nmax_size = 5', "unknown keys of .source_cache.: max_size"),
                           ('mode = "local"\ndir = "sessions"', "overlaps the recording session folder"),
                           ('mode = "local"\ndir = "work/a/cache"', "overlaps the work dir")]:
        path = write_conversion_batch(tmp_path, extra=f"[source_cache]\n{extra}\n")
        with pytest.raises(ValueError, match=message):
            load_conversion_batch(path)


def test_copy_reuse_and_release(tmp_path):
    source = _session(tmp_path, "s1", {"a.bin": 5000, "sub/b.bin": 300})
    lines = []
    cache = _cache(tmp_path, protected=[source], lines=lines)
    staged = cache.stage("s1", source)
    assert staged.copied and staged.origin == source.resolve() and staged.path == tmp_path / "cache" / "s1"
    assert (staged.path / "sub" / "b.bin").read_bytes() == (source / "sub" / "b.bin").read_bytes()
    assert folder_files(staged.path) == folder_files(source)  # sizes and modification times
    entry = json.loads((tmp_path / "cache" / "s1.json").read_text())
    assert entry["state"] == "complete" and entry["size"] == 5300 and entry["owner"]["pid"]

    # a paused session keeps its copy (local): reused on resume while the source is unchanged
    cache.release("s1", staged, keep=True)
    assert staged.path.is_dir() and json.loads((tmp_path / "cache" / "s1.json").read_text())["owner"] is None
    again = cache.stage("s1", source)
    assert again.path == staged.path and any("kept from before" in line for line in lines)
    cache.release("s1", again, keep=True)
    (source / "a.bin").write_bytes(b"changed")  # the source changed: copied again
    third = cache.stage("s1", source)
    assert (third.path / "a.bin").read_bytes() == b"changed"

    # done: the copy is deleted; the source is never touched
    cache.release("s1", third)
    assert not third.path.exists() and not (tmp_path / "cache" / "s1.json").exists() and (source / "a.bin").exists()


def test_memory_mode_never_keeps_copies(tmp_path):
    source = _session(tmp_path, "s1", {"a.bin": 1000})
    cache = _cache(tmp_path, mode="memory")
    staged = cache.stage("s1", source)
    cache.release("s1", staged, keep=True)
    assert not staged.path.exists()


def test_direct_and_too_big(tmp_path):
    source = _session(tmp_path, "s1", {"a.bin": 3000})
    staged = SourceCache(SourceCacheSettings()).stage("s1", source)
    assert not staged.copied and staged.path == source.resolve()
    lines = []
    staged = _cache(tmp_path, max_size_gb=2000 / GB, lines=lines).stage("s1", source)
    assert not staged.copied and staged.path == source.resolve()
    assert "more than the source cache holds" in lines[-1] and not (tmp_path / "cache" / "s1").exists()


def test_max_size_waits_for_space(tmp_path, monkeypatch):
    """Two sessions that do not fit together: the second waits until the first releases its copy."""
    monkeypatch.setattr(sc, "WAIT_SECONDS", 0.05)
    one = _session(tmp_path, "s1", {"a.bin": 3000})
    two = _session(tmp_path, "s2", {"a.bin": 3000})
    lines = []
    cache = _cache(tmp_path, max_size_gb=4000 / GB, lines=lines)
    first = cache.stage("s1", one)
    got = {}
    thread = threading.Thread(target=lambda: got.update(second=cache.stage("s2", two)))
    thread.start()
    time.sleep(0.3)
    assert "second" not in got and any("waiting for" in line for line in lines)
    cache.release("s1", first)
    thread.join(5)
    assert got["second"].copied and not first.path.exists()
    cache.release("s2", got["second"])

    # waiting ends when the runner stops
    first = cache.stage("s1", one)
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    assert cache.stage("s2", two, should_stop=stop.is_set) is None
    assert not (tmp_path / "cache" / "s2.json").exists()
    cache.release("s1", first)


def test_stale_copies_make_room(tmp_path):
    """The copy a gone runner left behind is deleted when its space is needed."""
    one = _session(tmp_path, "s1", {"a.bin": 3000})
    two = _session(tmp_path, "s2", {"a.bin": 3000})
    cache = _cache(tmp_path, max_size_gb=4000 / GB)
    cache.release("s1", cache.stage("s1", one), keep=True)  # kept, no owner
    staged = cache.stage("s2", two)
    assert staged.copied and not (tmp_path / "cache" / "s1").exists()


def test_runner_reads_the_copies(tmp_path):
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "local"\ndir = "cache"\nmax_size_gb = 1\n')
    lines: list[str] = []
    results = Runner(load_conversion_batch(path), echo=lines.append).run()
    assert results == {"a": "done", "b": "done"}, "\n".join(lines)
    conversion_batch = load_conversion_batch(path)
    assert _states(conversion_batch, "a") == {s: "done" for s in STEPS}
    assert sum("copying" in line for line in lines) == 2 and any("source cache: local" in line for line in lines)
    assert not list((tmp_path / "cache").glob("*.json"))  # deleted after each session
    job = conversion_batch.recording_session_job("a")
    assert read_record(job, "lfp")["argv"][2] == str(tmp_path / "cache" / "a")  # the steps read the copy
    provenance = json.loads((job.work_dir / "provenance" / "lfp.json").read_text())[-1]
    assert provenance["recording_session"] == str(job.recording_session_path)  # the original is recorded
    assert provenance["read_from"] == str(tmp_path / "cache" / "a")

    # nothing to run: nothing is copied; reading directly afterwards: the steps stay done
    lines.clear()
    Runner(load_conversion_batch(path), echo=lines.append).run()
    assert not any("copying" in line for line in lines)
    direct = SourceCacheSettings()
    lines.clear()
    assert Runner(load_conversion_batch(path), echo=lines.append, source_cache=direct).run() == {"a": "done", "b": "done"}
    assert not any(f" {s} {step}" in line for line in lines for s in "ab" for step in STEPS), lines


def test_run_options_override_the_file(tmp_path, monkeypatch):
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "local"\ndir = "cache"\n')
    seen = {}

    def fake_run(self, recording_session_ids=None, parallel=1):
        seen["settings"] = self.source_cache_settings
        return {}

    monkeypatch.setattr(Runner, "run", fake_run)
    main(["run", str(path), "--source-cache", "memory", "--source-cache-max-gb", "200"])
    assert seen["settings"].mode == "memory" and seen["settings"].dir == Path(sc.DEFAULT_MEMORY_DIR)
    assert seen["settings"].max_size_gb == 200
    main(["run", str(path), "--source-cache-dir", str(tmp_path / "other")])
    assert seen["settings"].mode == "local" and seen["settings"].dir == tmp_path / "other"
    main(["run", str(path), "--source-cache", "direct"])
    assert seen["settings"].mode == "direct"


def test_a_failed_copy_fails_only_its_session(tmp_path, monkeypatch):
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "local"\ndir = "cache"\n')
    real_copy = SourceCache._copy

    def copy(self, name, *args):
        if name == "a":
            raise OSError("No space left on device")
        return real_copy(self, name, *args)

    monkeypatch.setattr(SourceCache, "_copy", copy)
    lines: list[str] = []
    assert Runner(load_conversion_batch(path), echo=lines.append).run() == {"a": "failed", "b": "done"}
    assert any("a: copying to the source cache failed: No space left on device" in line for line in lines)


def test_memory_mode_only_on_linux(tmp_path, monkeypatch):
    """Elsewhere the file stays readable (status), but a run in memory mode is refused; local mode runs."""
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "memory"\n')
    monkeypatch.setattr(sc.sys, "platform", "win32")
    conversion_batch = load_conversion_batch(path)
    with pytest.raises(ValueError, match="needs Linux"):
        Runner(conversion_batch, echo=lambda _: None)
    local = SourceCacheSettings(SourceCacheMode.LOCAL, tmp_path / "cache")
    assert Runner(conversion_batch, echo=lambda _: None, source_cache=local).source_cache.settings.mode == "local"


def test_a_copy_that_cannot_be_deleted_yet(tmp_path, monkeypatch):
    """Windows cannot delete open files: the session still ends, the copy goes when its space is needed."""
    source = _session(tmp_path, "s1", {"a.bin": 1000})
    lines = []
    cache = _cache(tmp_path, lines=lines)
    staged = cache.stage("s1", source)

    def refuse(self, name):
        raise PermissionError("[WinError 32] The process cannot access the file")

    with monkeypatch.context() as m:
        m.setattr(SourceCache, "_remove", refuse)
        cache.release("s1", staged)
    assert "could not delete its copy" in lines[-1]
    assert json.loads((tmp_path / "cache" / "s1.json").read_text())["owner"] is None  # stale: deleted later
    cache.release("s1", cache.stage("s1", source))  # reused, then deleted
    assert not staged.path.exists()
