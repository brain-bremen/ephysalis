"""``dataset.toml``: many sessions processed alike (:mod:`ephysalis.dataset.runner`).

Example (paths are relative to ``root``, which is relative to the file; default: the file's folder)::

    schema_version = 1
    name = "V1 array recordings of monkey B"
    root = ".."
    work_dir = "derivatives/ephysalis/{session}"
    state_dir = "derivatives/ephysalis"            # pause file and status table (default: <root>/.ephysalis)
    steps = ["align-clocks", "extract-trials", "extract-events", "lfp", "export"]

    [defaults]                                     # session.toml content of every session
    schema_version = 8
    lab = "Kreiter lab"
    [defaults.aux_channels.digital]
    trial_start = 0

    [step_options.lfp]                             # options of a step, as on its command line
    n_jobs = 4
    [step_options.export]
    output_format = "dh5"
    per_run = true

    [[sessions]]
    id = "sub-b_ses-20101207"                      # {session} in work_dir and in step options
    path = "sourcedata/sub-b/ses-20101207"
    config = "code/sessions/sub-b_ses-20101207/session.toml"   # overrides of the defaults (optional)
    [sessions.overrides]                           # inline overrides (optional), after the file
    task = "shapetracking"
    [sessions.step_options.export]                 # this session's step options (optional)
    session_label = "20101207"

A session's config is the defaults, overridden by its ``config`` file, overridden by its ``overrides``:
tables are merged key by key, everything else (values, arrays such as ``runs``) is replaced. Each part
is migrated from its own ``schema_version`` (``overrides``: that of the defaults) before merging.
Relative probe geometry files are relative to the file that names them (the defaults and inline
overrides: the dataset file).
"""

from __future__ import annotations

import argparse
import copy
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ephysalis.session.config import (
    MIGRATIONS,
    SCHEMA_VERSION,
    SessionConfig,
    session_config_from_dict,
)
from ephysalis.session.schema import migrate

DATASET_SCHEMA_VERSION = 1
DATASET_MIGRATIONS: dict[int, Any] = {}
SESSION_PLACEHOLDER = "{session}"
# steps in the order they run and the steps whose outputs each one reads
STEP_INPUTS: dict[str, tuple[str, ...]] = {
    "align-clocks": (),
    "calibrate": (),
    "extract-trials": ("align-clocks",),
    "extract-events": ("align-clocks",),
    "extract-behavior": ("align-clocks",),
    "lfp": ("align-clocks", "calibrate"),
    "quality": ("lfp",),
    "csd": ("quality",),
    "mua": ("calibrate",),
    "sort": ("align-clocks", "calibrate"),
    "export": ("align-clocks", "calibrate", "extract-trials", "extract-events", "extract-behavior", "lfp", "quality",
               "csd", "mua", "sort"),
}
# options the runner sets itself
_RUNNER_OPTIONS = {"config", "work_dir", "format", "runs", "force", "sessions"}


@dataclass
class DatasetSession:
    id: str
    path: Path
    work_dir: Path
    config: SessionConfig
    config_files: list[Path]  # the files the config was made from (dataset file, override file)
    referenced_files: list[Path]  # files the config names (probe geometry)
    step_options: dict[str, dict[str, Any]]
    include: bool = True


@dataclass
class Dataset:
    path: Path
    name: str
    root: Path
    state_dir: Path
    steps: list[str]
    sessions: list[DatasetSession] = field(default_factory=list)

    def session(self, session_id: str) -> DatasetSession:
        for s in self.sessions:
            if s.id == session_id:
                return s
        raise KeyError(f"{self.path}: no session {session_id!r}")


def merge_tables(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``override`` on ``base``: tables merged key by key, other values (and arrays) replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_tables(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_geometry_files(data: dict[str, Any], folder: Path) -> dict[str, Any]:
    """Relative ``probes.<id>.geometry.file`` of ``data`` as absolute paths (relative to ``folder``)."""
    data = copy.deepcopy(data)
    for probe in (data.get("probes") or {}).values():
        geometry = probe.get("geometry") if isinstance(probe, dict) else None
        if isinstance(geometry, dict) and geometry.get("file") and not Path(geometry["file"]).is_absolute():
            geometry["file"] = str((folder / geometry["file"]).resolve())
    return data


def _session_part(data: dict[str, Any], version: int, folder: Path, source: object) -> dict[str, Any]:
    data = migrate(dict(data, schema_version=data.get("schema_version", version)), SCHEMA_VERSION, MIGRATIONS,
                   source=source)
    return _resolve_geometry_files(data, folder)


def _check_step_options(step: str, options: dict[str, Any], source: str) -> None:
    from ephysalis.provenance import _step_parser

    known = {a.dest for a in _step_parser(step)._actions if a.option_strings} - {"help"}
    unknown = set(options) - known
    if unknown:
        raise ValueError(f"{source}: unknown options of {step}: {', '.join(sorted(unknown))} "
                         f"(known: {', '.join(sorted(known - _RUNNER_OPTIONS))})")
    reserved = set(options) & _RUNNER_OPTIONS
    if reserved:
        raise ValueError(f"{source}: {', '.join(sorted(reserved))} of {step} are set by the runner")


def step_argv(step: str, options: dict[str, Any]) -> list[str]:
    """Command line options of ``step`` from ``options`` (``{dest: value}``, e.g. ``{"n_jobs": 4}``)."""
    from ephysalis.provenance import _step_parser

    actions = {a.dest: a for a in _step_parser(step)._actions if a.option_strings}
    argv: list[str] = []
    for dest, value in options.items():
        action = actions[dest]
        flag = max(action.option_strings, key=len)
        if value is None or value is False:
            continue
        if value is True:
            if not isinstance(action, argparse._StoreTrueAction):
                raise ValueError(f"{step} --{dest}: takes a value, not true")
            argv.append(flag)
        elif isinstance(value, list):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path).resolve()
    data = migrate(tomllib.loads(path.read_text(encoding="utf-8")), DATASET_SCHEMA_VERSION, DATASET_MIGRATIONS,
                   source=path)
    known = {"schema_version", "name", "root", "work_dir", "state_dir", "steps", "defaults", "step_options", "sessions"}
    if unknown := set(data) - known:
        raise ValueError(f"{path}: unknown keys {', '.join(sorted(unknown))}")
    root = (path.parent / data.get("root", ".")).resolve()
    steps = data.get("steps", list(STEP_INPUTS))
    if bad := [s for s in steps if s not in STEP_INPUTS]:
        raise ValueError(f"{path}: unknown steps {', '.join(bad)} (known: {', '.join(STEP_INPUTS)})")
    steps = [s for s in STEP_INPUTS if s in steps]  # in dependency order
    step_options = data.get("step_options", {})
    for step, options in step_options.items():
        if step not in STEP_INPUTS:
            raise ValueError(f"{path}: step_options of unknown step {step}")
        _check_step_options(step, options, str(path))
    raw_defaults = data.get("defaults", {})
    defaults_version = raw_defaults.get("schema_version", SCHEMA_VERSION)
    defaults = _session_part(raw_defaults, defaults_version, path.parent, f"{path} [defaults]")
    work_template = data.get("work_dir", "{session}/ephysalis")
    dataset = Dataset(path=path, name=data.get("name", path.parent.name), root=root,
                      state_dir=(root / data.get("state_dir", ".ephysalis")).resolve(), steps=steps)
    seen = set()
    for entry in data.get("sessions", []):
        session_id = entry.get("id")
        if not session_id or "path" not in entry:
            raise ValueError(f"{path}: every [[sessions]] needs id and path ({entry})")
        if session_id in seen:
            raise ValueError(f"{path}: session {session_id} twice")
        seen.add(session_id)
        if unknown := set(entry) - {"id", "path", "config", "overrides", "step_options", "include"}:
            raise ValueError(f"{path} session {session_id}: unknown keys {', '.join(sorted(unknown))}")
        merged, files = defaults, [path]
        if "config" in entry:
            config_file = (root / entry["config"]).resolve()
            raw = tomllib.loads(config_file.read_text(encoding="utf-8"))
            merged = merge_tables(merged, _session_part(raw, raw.get("schema_version", 1), config_file.parent,
                                                        config_file))
            files.append(config_file)
        if "overrides" in entry:
            merged = merge_tables(merged, _session_part(entry["overrides"], defaults_version, path.parent,
                                                        f"{path} session {session_id} overrides"))
        merged.setdefault("session_id", session_id)
        config = session_config_from_dict(merged, source=f"{path} session {session_id}")
        options = merge_tables(step_options, entry.get("step_options", {}))
        for step, step_opts in entry.get("step_options", {}).items():
            if step not in STEP_INPUTS:
                raise ValueError(f"{path} session {session_id}: step_options of unknown step {step}")
            _check_step_options(step, step_opts, f"{path} session {session_id}")
        options = {step: {k: _fill(v, session_id) for k, v in opts.items()} for step, opts in options.items()}
        session_path = (root / entry["path"]).resolve()
        referenced = [Path(p.geometry.file) for p in config.probes.values() if p.geometry and p.geometry.file]
        dataset.sessions.append(DatasetSession(
            id=session_id, path=session_path,
            work_dir=(root / _fill(work_template, session_id)).resolve(),
            config=config, config_files=files, referenced_files=referenced, step_options=options,
            include=entry.get("include", True)))
    return dataset


def _fill(value: Any, session_id: str) -> Any:
    """``value`` with :data:`SESSION_PLACEHOLDER` replaced by the session id (also inside lists)."""
    if isinstance(value, str):
        return value.replace(SESSION_PLACEHOLDER, session_id)
    if isinstance(value, list):
        return [_fill(v, session_id) for v in value]
    return value
