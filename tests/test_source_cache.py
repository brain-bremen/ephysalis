"""[source_cache] of a conversion batch: the steps read the session folders directly, or a copy on a local disk,
below a size limit (ephysalis.conversion_batch.source_cache)."""

import json
import os
import socket
import subprocess
import sys
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
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "local"\ndir = "cache"\nmax_size_gb = 500\n')
    cache = load_conversion_batch(path).source_cache
    assert cache.dir == (tmp_path / "cache").resolve() and cache.max_size_gb == 500.0 and cache.reserve_bytes == 10 * GB
    for extra, message in [('mode = "memory"', "is not one of direct, local"),
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
    from ephysalis.provenance import input_fingerprint

    assert provenance["inputs"] == input_fingerprint(job.recording_session_path)  # same files, sizes, hashes

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
    main(["run", str(path), "--source-cache-max-gb", "200"])
    assert seen["settings"].mode == "local" and seen["settings"].max_size_gb == 200
    main(["run", str(path), "--source-cache-dir", str(tmp_path / "other")])
    assert seen["settings"].mode == "local" and seen["settings"].dir == tmp_path / "other"
    main(["run", str(path), "--source-cache", "direct"])
    assert seen["settings"].mode == "direct"


def test_a_failed_copy_fails_only_its_session(tmp_path, monkeypatch):
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "local"\ndir = "cache"\n')
    real_copyfile = sc.shutil.copyfile

    def copyfile(src, dst, *args, **kwargs):
        if Path(dst).parent.name.startswith("a.partial") or "a.partial" in Path(dst).parts:
            Path(dst).write_bytes(b"part")
            raise OSError("No space left on device")
        return real_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(sc.shutil, "copyfile", copyfile)
    lines: list[str] = []
    assert Runner(load_conversion_batch(path), echo=lines.append).run() == {"a": "failed", "b": "done"}
    assert any("a: copying to the source cache failed: No space left on device" in line for line in lines)
    assert not list((tmp_path / "cache").glob("a*"))  # no partial copy, no entry


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


def test_prefetch_copies_ahead(tmp_path):
    source = _session(tmp_path, "s1", {"a.bin": 3000})
    lines = []
    cache = _cache(tmp_path, lines=lines)
    assert cache.prefetch("s1", source) and cache.prefetched == {"s1"}
    assert any("(prefetch)" in line for line in lines)
    assert cache.prefetch("s1", source)  # there already: not copied again
    assert sum("copying" in line for line in lines) == 1
    staged = cache.stage("s1", source)  # the session takes the prefetched copy
    assert staged.copied and "(prefetched)" in lines[-1] and cache.prefetched == set()
    cache.release("s1", staged)
    assert not staged.path.exists()


def test_prefetch_only_when_it_fits_and_running_sessions_go_first(tmp_path):
    one = _session(tmp_path, "s1", {"a.bin": 3000})
    two = _session(tmp_path, "s2", {"a.bin": 3000})
    three = _session(tmp_path, "s3", {"a.bin": 3000})
    lines = []
    cache = _cache(tmp_path, max_size_gb=7000 / GB, lines=lines)
    first = cache.stage("s1", one)
    assert cache.prefetch("s2", two)
    t0 = time.time()
    assert not cache.prefetch("s3", three) and time.time() - t0 < 1  # does not fit: no copy, no waiting
    cache.release("s1", first)
    staged = cache.stage("s3", three)  # fits next to the prefetched s2
    assert staged.copied and cache.prefetched == {"s2"}
    # a running session that needs the space deletes prefetched copies (here: s2 for s1)
    again = cache.stage("s1", one)
    assert again.copied and not (tmp_path / "cache" / "s2").exists() and cache.prefetched == set()
    assert any("s1: deleting the prefetched copy of s2" in line for line in lines)


def test_a_session_waits_for_its_prefetch(tmp_path, monkeypatch):
    source = _session(tmp_path, "s1", {"a.bin": 3000, "b.bin": 3000})
    lines = []
    cache = _cache(tmp_path, lines=lines)
    real_copyfile = sc.shutil.copyfile

    def slow_copyfile(*args, **kwargs):
        time.sleep(0.3)
        return real_copyfile(*args, **kwargs)

    monkeypatch.setattr(sc.shutil, "copyfile", slow_copyfile)
    thread = threading.Thread(target=cache.prefetch, args=("s1", source))
    thread.start()
    time.sleep(0.1)
    staged = cache.stage("s1", source)  # while the prefetch copies: waits for it, does not copy again
    thread.join()
    assert staged.copied and sum("copying" in line for line in lines) == 1 and "(prefetched)" in lines[-1]


def test_prefetch_stops(tmp_path):
    source = _session(tmp_path, "s1", {"a.bin": 3000})
    cache = _cache(tmp_path)
    assert not cache.prefetch("s1", source, should_stop=lambda: True)
    assert not list((tmp_path / "cache").glob("s1*"))


def test_runner_prefetches_the_next_session(tmp_path, monkeypatch):
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "local"\ndir = "cache"\n')
    real_run_step = Runner.run_step

    def run_step(self, job, step, *args, **kwargs):  # session a computes long enough for b to be prefetched
        if job.recording_session_id == "a":
            time.sleep(0.5)
        return real_run_step(self, job, step, *args, **kwargs)

    monkeypatch.setattr(Runner, "run_step", run_step)
    lines: list[str] = []
    assert Runner(load_conversion_batch(path), echo=lines.append).run(parallel=1) == {"a": "done", "b": "done"}
    assert any("b: copying" in line and "(prefetch)" in line for line in lines), lines
    assert any("b: reading the copy" in line and "(prefetched)" in line for line in lines), lines
    assert sum("copying" in line for line in lines) == 2 and not list((tmp_path / "cache").glob("*.json"))


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_a_change_of_the_same_size_is_noticed(tmp_path):
    """Reuse needs the same size and modification time of every file."""
    source = _session(tmp_path, "s1", {"a.bin": 3000})
    cache = _cache(tmp_path)
    cache.release("s1", cache.stage("s1", source), keep=True)
    (source / "a.bin").write_bytes(b"y" * 3000)  # same size, new content and time
    os.utime(source / "a.bin", ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    staged = cache.stage("s1", source)
    assert (staged.path / "a.bin").read_bytes() == b"y" * 3000


def test_a_crashed_copy_is_never_read(tmp_path):
    """A runner killed while copying leaves a partial copy and an entry in state copying: deleted, copied anew."""
    source = _session(tmp_path, "s1", {"a.bin": 3000})
    cache = _cache(tmp_path)
    (tmp_path / "cache" / "s1.partial").mkdir(parents=True)
    (tmp_path / "cache" / "s1.partial" / "a.bin").write_bytes(b"half")
    (tmp_path / "cache" / "s1.json").write_text(json.dumps({
        "source": str(source.resolve()), "size": 3000, "files": folder_files(source), "state": "copying",
        "owner": {"host": socket.gethostname(), "pid": _dead_pid()}}))
    staged = cache.stage("s1", source)
    assert (staged.path / "a.bin").read_bytes() == (source / "a.bin").read_bytes()
    assert not (tmp_path / "cache" / "s1.partial").exists()
    # also a complete-looking folder without a complete entry is not used
    cache.release("s1", staged, keep=True)
    entry = json.loads((tmp_path / "cache" / "s1.json").read_text())
    (tmp_path / "cache" / "s1.json").write_text(json.dumps({**entry, "state": "copying"}))
    (staged.path / "a.bin").write_bytes(b"x" * 3000)  # the content of an unfinished copy
    assert (cache.stage("s1", source).path / "a.bin").read_bytes() == (source / "a.bin").read_bytes()


def test_the_copies_of_another_live_runner(tmp_path, monkeypatch):
    """Not taken over, not deleted, and their space counts."""
    monkeypatch.setattr(sc, "WAIT_SECONDS", 0.05)
    one = _session(tmp_path, "s1", {"a.bin": 3000})
    two = _session(tmp_path, "s2", {"a.bin": 3000})
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        theirs = _cache(tmp_path, max_size_gb=4000 / GB)
        theirs.owner = {"host": socket.gethostname(), "pid": other.pid}  # as if made by the other runner
        theirs.stage("s1", one)
        ours = _cache(tmp_path, max_size_gb=4000 / GB)
        with pytest.raises(RuntimeError, match="used by another runner"):
            ours.stage("s1", one)
        stop = threading.Event()
        threading.Timer(0.3, stop.set).start()
        assert ours.stage("s2", two, should_stop=stop.is_set) is None  # waits: their copy holds the space
        assert not ours.prefetch("s2", two)
        assert (tmp_path / "cache" / "s1" / "a.bin").exists()
    finally:
        other.kill()
        other.wait()
    assert ours.stage("s2", two).copied  # their runner is gone: its copy makes room


def test_never_deletes_a_source_folder(tmp_path):
    """Even a symbolic link in the cache dir that points to a session folder is not followed when deleting."""
    one = _session(tmp_path, "s1", {"a.bin": 3000})
    two = _session(tmp_path, "s2", {"a.bin": 3000})
    cache = _cache(tmp_path, max_size_gb=4000 / GB, protected=[one, two])
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "s1").symlink_to(one, target_is_directory=True)
    (tmp_path / "cache" / "s1.json").write_text(json.dumps({"size": 3000, "state": "complete", "owner": None}))
    with pytest.raises(RuntimeError, match="refusing to delete"):
        cache.stage("s2", two)  # needs the space of the (stale) entry s1
    assert (one / "a.bin").read_bytes() and (two / "a.bin").exists()


def test_concurrent_copies_stay_below_max_size(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "WAIT_SECONDS", 0.02)
    sources = [_session(tmp_path, f"s{i}", {"a.bin": 3000}) for i in range(4)]
    cache = _cache(tmp_path, max_size_gb=7000 / GB)
    real_copyfile = sc.shutil.copyfile
    peak = []

    def copyfile(src, dst, *args, **kwargs):
        time.sleep(0.05)
        result = real_copyfile(src, dst, *args, **kwargs)
        peak.append(sum(p.stat().st_size for p in (tmp_path / "cache").rglob("*.bin")))
        return result

    monkeypatch.setattr(sc.shutil, "copyfile", copyfile)

    def session(i):
        staged = cache.stage(f"s{i}", sources[i])
        time.sleep(0.1)  # the steps
        cache.release(f"s{i}", staged)

    threads = [threading.Thread(target=session, args=(i,)) for i in range(4)]
    threads.append(threading.Thread(target=cache.prefetch, args=("s3", sources[3])))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert peak and max(peak) <= 7000 and not list((tmp_path / "cache").glob("s*"))


def test_a_stop_during_the_copy(tmp_path, monkeypatch):
    path = write_conversion_batch(tmp_path, extra='[source_cache]\nmode = "local"\ndir = "cache"\n')
    runner = Runner(load_conversion_batch(path), echo=lambda _: None)
    real_copyfile = sc.shutil.copyfile

    def copyfile(src, dst, *args, **kwargs):
        runner.interrupt()  # Ctrl+C while the first file is copied
        return real_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(sc.shutil, "copyfile", copyfile)
    results = runner.run(parallel=1)
    assert set(results.values()) == {"interrupted"} and not list((tmp_path / "cache").glob("*.partial"))
    assert not [p for p in (tmp_path / "cache").glob("*.json")
                if json.loads(p.read_text()).get("state") != "complete"]
    monkeypatch.undo()
    assert Runner(load_conversion_batch(path), echo=lambda _: None).run() == {"a": "done", "b": "done"}
