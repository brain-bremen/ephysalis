"""Processing record: the config, options and software versions the outputs were made with.

The goal is reproducible processing. Every pipeline step appends a record of its invocation
to ``<work dir>/provenance/<step>.json`` (a list, as a step can run several times, e.g. one
probe at a time): the ``session.toml`` it used (with its schema version), its command line
and options, the complete Python environment (all installed packages; the git commit of
ephyscon when it runs from a checkout) and a fingerprint of the input files (size of every
file, SHA-256 of the small ones, i.e. the metadata files). The export collects these records, the sorter settings of every probe
(``sorting/<probe>/segments.json``) and its own options into one processing record and
embeds it as JSON in the output file:

- NWB: ``scratch/ephyscon_processing`` (text dataset)
- DH5: attribute ``Processing`` of the ``/Operations/nnn_ephyscon_export`` group (the DH5
  specification keeps information about an operation in its group's attributes)

:func:`read_processing_record` reads it back, :func:`record_config` returns the
``session.toml`` of a step as a :class:`SessionConfig`, :func:`replay_commands` the commands
that redo the processing (``ephyscon reproduce``) and :func:`compare_environment` /
:func:`compare_inputs` what differs from the recorded environment and inputs.

Reproducible are: sync, timing, LFP, the SortingAnalyzer (fixed seeds) and the export.
Kilosort on a GPU is not bit-exact between runs (nor between GPUs and CUDA versions).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from ephyscon.session import Session
    from ephyscon.session_config import SessionConfig

RECORD_SCHEMA_VERSION = 1
HASH_MAX_BYTES = 1_000_000  # input files up to this size are hashed (metadata), larger ones by size
# steps in pipeline order, as replayed by ``ephyscon reproduce``
REPLAY_STEPS = ("sync", "timing", "lfp", "sort", "export")
# options not replayed: outputs and caches outside the work dir
_NOT_REPLAYED = {"output", "bids", "existing", "cache_dir", "subject", "session_label", "dataset_name"}
NWB_SCRATCH_NAME = "ephyscon_processing"
DH5_ATTRIBUTE_NAME = "Processing"
DH5_OPERATION_NAME = "ephyscon_export"
PACKAGES = (
    "ephyscon", "spikeinterface", "probeinterface", "kilosort", "torch", "numpy", "scipy", "pynwb", "hdmf",
    "dh5io", "vstim", "open-ephys-python-tools",
)


def software_versions() -> dict[str, str]:
    """Versions of Python and of the installed :data:`PACKAGES`."""
    versions = {"python": sys.version.split()[0]}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return versions


def ephyscon_source() -> dict[str, Any] | None:
    """Git commit (and uncommitted changes) of ephyscon when installed from a checkout."""
    try:
        direct_url = json.loads(metadata.distribution("ephyscon").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        return None
    if not direct_url.get("url", "").startswith("file:"):
        return None
    folder = unquote(urlparse(direct_url["url"]).path).lstrip("/") if sys.platform == "win32" else \
        unquote(urlparse(direct_url["url"]).path)

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", "-c", "safe.directory=*", "-C", folder, *args], capture_output=True,
                                    text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"path": folder}
    changes = git("status", "--porcelain", "--untracked-files=no")
    return {"path": folder, "commit": commit, "dirty": bool(changes), "changed_files": (changes or "").splitlines()}


def environment() -> dict[str, Any]:
    """Python, platform, all installed packages and the ephyscon source."""
    import platform

    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            packages[name.lower()] = dist.version
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": dict(sorted(packages.items())), "ephyscon_source": ephyscon_source()}


def input_fingerprint(session_path: Path, exclude: list[Path] | None = None) -> dict[str, dict[str, Any]]:
    """``{relative path: {size, sha256}}`` of the files of a session (``sha256`` for files up
    to :data:`HASH_MAX_BYTES`), leaving out the folders ``exclude`` (e.g. the work dir)."""
    excluded = [p.resolve() for p in exclude or []]
    files = {}
    for path in sorted(session_path.rglob("*")):
        if not path.is_file() or any(path.resolve().is_relative_to(e) for e in excluded):
            continue
        size = path.stat().st_size
        entry: dict[str, Any] = {"size": size}
        if size <= HASH_MAX_BYTES:
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        files[path.relative_to(session_path).as_posix()] = entry
    return files


def step_record(step: str, session: Session, config: SessionConfig | None, config_path: Path | None,
                options: dict[str, Any], argv: list[str] | None = None,
                work_dir: Path | None = None) -> dict[str, Any]:
    """Record of one invocation of ``step`` (``config`` None: defaults from the recording)."""
    return {
        "step": step,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": ["ephyscon", *argv] if argv is not None else None,
        "session": str(session.path),
        "config_file": str(config_path) if config_path else None,
        "config": config.model_dump(mode="json", exclude_none=True) if config else None,
        "options": options,
        "software": software_versions(),
        "environment": environment(),
        "inputs": input_fingerprint(session.path, [work_dir] if work_dir else None),
    }


def provenance_dir(work_dir: Path) -> Path:
    return work_dir / "provenance"


def write_step_record(work_dir: Path, record: dict[str, Any]) -> Path:
    """Append the record of a step invocation (:func:`step_record`) to ``provenance/<step>.json``."""
    path = provenance_dir(work_dir) / f"{record['step']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    records.append(record)
    path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    return path


def step_records(work_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """``{step: [records]}`` of the work dir."""
    folder = provenance_dir(work_dir)
    if not folder.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}


def processing_record(work_dir: Path, export: dict[str, Any]) -> dict[str, Any]:
    """Processing record of an output file: the steps' records, the sorter settings of every
    probe and the export's own record ``export`` (see :func:`step_record`)."""
    sorting = {}
    for segments in sorted((work_dir / "sorting").glob("*/segments.json")):
        info = json.loads(segments.read_text(encoding="utf-8"))
        sorting[segments.parent.name] = {k: info.get(k) for k in ("sorter", "sorter_params", "channel_ids")}
    return {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "work_dir": str(work_dir),
        "export": export,
        "steps": step_records(work_dir),
        "sorting": sorting,
    }


def read_processing_record(path: str | Path) -> dict[str, Any] | None:
    """The processing record embedded in an NWB or DH5 file (None if there is none)."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as file:
        if path.suffix.lower() == ".nwb":
            node = file.get(f"scratch/{NWB_SCRATCH_NAME}")
            if isinstance(node, h5py.Group):  # ScratchData of an array
                node = node.get("data")
            if node is None:
                return None
            text = node[()]
        else:
            operations = file.get("Operations")
            groups = [g for name, g in (operations.items() if operations else [])
                      if name.endswith(DH5_OPERATION_NAME) and DH5_ATTRIBUTE_NAME in g.attrs]
            if not groups:
                return None
            text = groups[-1].attrs[DH5_ATTRIBUTE_NAME]
    return json.loads(text.decode() if isinstance(text, bytes) else str(text))


def record_config(record: dict[str, Any], step: str = "export") -> SessionConfig | None:
    """``session.toml`` used by ``step`` (its last invocation), upgraded to the current schema."""
    from ephyscon.session_config import session_config_from_dict

    entry = record["export"] if step == "export" else (record["steps"].get(step) or [None])[-1]
    if entry is None or entry.get("config") is None:
        return None
    return session_config_from_dict(entry["config"], source=f"processing record ({step})")


def replay_invocations(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The step invocations that made an output, in the order they ran: all recorded
    invocations of sync, timing, lfp and sort, then the export of the file."""
    steps = [r for step in REPLAY_STEPS[:-1] for r in record["steps"].get(step, [])]
    return sorted(steps, key=lambda r: r["time"]) + [record["export"]]


def _step_parser(step: str) -> Any:
    import argparse

    from ephyscon.cli.session_commands import add_session_commands

    parser = argparse.ArgumentParser(prog="ephyscon")
    sub = parser.add_subparsers(dest="command")
    add_session_commands(sub)
    return sub.choices[step]


def replay_argv(invocation: dict[str, Any], session: Path, work_dir: Path, config: Path | None) -> list[str]:
    """Command line (without ``ephyscon``) that redoes ``invocation`` into ``work_dir``, with
    the session at ``session`` and the recorded config saved at ``config``."""
    options = dict(invocation["options"], work_dir=str(work_dir), config=str(config) if config else None)
    argv = [invocation["step"], str(session)]
    for action in _step_parser(invocation["step"])._actions:
        if not action.option_strings or action.dest in _NOT_REPLAYED or action.dest == "help":
            continue
        value = options.get(action.dest, action.default)
        if value is None or value is False or value == action.default:
            continue
        flag = max(action.option_strings, key=len)
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv


def replay_commands(record: dict[str, Any], session: Path, work_dir: Path) -> list[tuple[list[str], dict | None]]:
    """``[(argv, config)]`` that redo the processing of an output: each invocation's command
    line and the raw ``session.toml`` it used (None: defaults from the recording). The
    caller writes the configs; :func:`write_replay_configs` does so."""
    return [(replay_argv(inv, session, work_dir, None), inv.get("config")) for inv in replay_invocations(record)]


def write_replay_configs(record: dict[str, Any], session: Path, work_dir: Path) -> list[list[str]]:
    """Write the configs of the recorded steps to ``work_dir`` (``session.toml``, or
    ``session_<n>.toml`` when the steps used different ones) and return the commands that
    redo the processing with them."""
    from ephyscon.session_config import save_session_config, session_config_from_dict

    invocations = replay_invocations(record)
    configs: list[dict] = []
    for inv in invocations:
        if inv.get("config") is not None and inv["config"] not in configs:
            configs.append(inv["config"])
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, data in enumerate(configs):
        path = work_dir / ("session.toml" if len(configs) == 1 else f"session_{i + 1}.toml")
        save_session_config(session_config_from_dict(data, source="processing record"), path)
        paths.append(path)
    return [replay_argv(inv, session, work_dir, paths[configs.index(inv["config"])] if inv.get("config") else None)
            for inv in invocations]


def compare_environment(recorded: dict[str, Any], current: dict[str, Any] | None = None) -> list[str]:
    """Differences between a recorded environment and the current one (or ``current``)."""
    current = current or environment()
    differences = []
    old, new = recorded.get("packages", {}), current.get("packages", {})
    for name in sorted(set(old) | set(new)):
        if old.get(name) != new.get(name):
            differences.append(f"{name}: {old.get(name, 'not installed')} -> {new.get(name, 'not installed')}")
    if recorded.get("python") != current.get("python"):
        differences.append(f"python: {recorded.get('python')} -> {current.get('python')}")
    old_src, new_src = recorded.get("ephyscon_source") or {}, current.get("ephyscon_source") or {}
    if old_src.get("commit") != new_src.get("commit"):
        differences.append(f"ephyscon commit: {old_src.get('commit')} -> {new_src.get('commit')}")
    if old_src.get("dirty"):
        differences.append("ephyscon had uncommitted changes when the output was made")
    return differences


def compare_inputs(recorded: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]]) -> list[str]:
    """Input files that are missing, new or changed (size or SHA-256) compared to ``recorded``."""
    differences = [f"missing: {p}" for p in sorted(set(recorded) - set(current))]
    differences += [f"new: {p}" for p in sorted(set(current) - set(recorded))]
    differences += [f"changed: {p}" for p in sorted(set(recorded) & set(current)) if recorded[p] != current[p]]
    return differences
