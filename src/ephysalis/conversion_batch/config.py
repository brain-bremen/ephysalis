"""``conversion_batch.toml``: many recording sessions processed alike (:mod:`ephysalis.conversion_batch.runner`).

Example (paths are relative to ``root``, which is relative to the file; default: the file's folder)::

    schema_version = 2
    conversion_batch_name = "V1 array recordings of monkey B"
    root = ".."
    work_dir = "derivatives/ephysalis/{recording_session}"
    state_dir = "derivatives/ephysalis"            # pause file and status table (default: <root>/.ephysalis)
    steps = ["align-clocks", "extract-events", "lfp", "export"]
    recording_session_root = "sourcedata"          # the folders below are relative to it (default: root)
    recording_session_config_file = "code/sessions/{recording_session}/session.toml"   # optional

    [recording_sessions]                           # folder -> its recording session folders
    sub-b = [
        "ses-20101207",
        "ses-20101209",
    ]

    [defaults]                                     # session.toml content of every recording session
    schema_version = 10
    lab = "Kreiter lab"
    [defaults.aux_channels.digital]
    trial_start = { bit = 0, description = "trial start" }

    [step_options.lfp]                             # options of a step, as on its command line
    n_jobs = 4
    [step_options.export]
    output_format = "dh5"
    session_label = "{session_label}"

    [recording_session_exceptions."sub-b/ses-20101209"]   # only for sessions that differ (all keys optional)
    recording_session_config_file = "code/sessions/b_20101209.toml"
    include = false
    [recording_session_exceptions."sub-b/ses-20101209".session_config]   # session.toml content, after the file
    task = "shapetracking"
    [recording_session_exceptions."sub-b/ses-20101209".step_options.export]
    per_run = false

A recording session is a folder: ``<recording_session_root>/<folder>/<session folder>`` (folder ``"."``: directly
in the root). Its id is the folder path with ``/`` as ``_`` (``sub-b_ses-20101207``); the id names its work dir,
its row in the status table and its ``--recording-sessions`` selection. Placeholders in ``work_dir``,
``recording_session_config_file`` and step options: ``{recording_session}`` = the id, ``{subject}`` and
``{session_label}`` = its parts if the id is ``sub-<subject>_ses-<session_label>``. The input format of each
recording session is detected from its files (the input format plugins), or set by ``input_format`` in its config.

A recording session's config is the defaults, overridden by its ``recording_session_config_file`` (if the
template is given, the file must exist), overridden by its exception's ``session_config``: tables are merged key
by key, everything else (values, arrays such as ``runs``) is replaced. Every part must have the current
``schema_version`` (or none). Relative probe geometry files are
relative to the file that names them (the defaults and ``session_config``: the conversion batch file).
"""

from __future__ import annotations

import argparse
import copy
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ephysalis.recording_session.config import (
    MIGRATIONS,
    SCHEMA_VERSION,
    RecordingSessionConfig,
    recording_session_config_from_dict,
)
from ephysalis.recording_session.schema import migrate

CONVERSION_BATCH_SCHEMA_VERSION = 2


CONVERSION_BATCH_MIGRATIONS: dict[int, Any] = {}  # older versions are not read
RECORDING_SESSION_PLACEHOLDER = "{recording_session}"
BIDS_RECORDING_SESSION_ID = re.compile(r"^sub-(?P<subject>[A-Za-z0-9]+)_ses-(?P<session_label>[A-Za-z0-9]+)$")
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
    "spike-sort": ("align-clocks", "calibrate"),
    "export": ("align-clocks", "calibrate", "extract-trials", "extract-events", "extract-behavior", "lfp", "quality",
               "csd", "mua", "spike-sort"),
}
# options the runner sets itself
_RUNNER_OPTIONS = {"config", "work_dir", "format", "runs", "force", "recording_session_jobs"}


@dataclass
class RecordingSessionJob:
    recording_session_id: str  # {session} in work_dir and in step options
    recording_session_path: Path  # the recording's session folder
    work_dir: Path
    recording_session_config: RecordingSessionConfig  # defaults < session file < inline overrides
    config_files: list[Path]  # the files the config was made from (conversion batch file, override file)
    referenced_files: list[Path]  # files the config names (probe geometry)
    step_options: dict[str, dict[str, Any]]
    include: bool = True


@dataclass
class ConversionBatch:
    conversion_batch_file: Path  # conversion_batch.toml
    conversion_batch_name: str
    root: Path  # relative paths of the conversion batch file are relative to it
    state_dir: Path
    steps: list[str]
    recording_session_jobs: list[RecordingSessionJob] = field(default_factory=list)

    def recording_session_job(self, recording_session_id: str) -> RecordingSessionJob:
        for recording_session_job in self.recording_session_jobs:
            if recording_session_job.recording_session_id == recording_session_id:
                return recording_session_job
        raise KeyError(f"{self.conversion_batch_file}: no session {recording_session_id!r}")


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


def _recording_session_part(data: dict[str, Any], version: int, folder: Path, source: object) -> dict[str, Any]:
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


def load_conversion_batch(path: str | Path) -> ConversionBatch:
    path = Path(path).resolve()
    data = migrate(tomllib.loads(path.read_text(encoding="utf-8")), CONVERSION_BATCH_SCHEMA_VERSION,
                   CONVERSION_BATCH_MIGRATIONS, missing=CONVERSION_BATCH_SCHEMA_VERSION, source=path)
    known = {"schema_version", "conversion_batch_name", "root", "work_dir", "state_dir", "steps", "defaults",
             "step_options", "recording_session_root", "recording_session_config_file", "recording_sessions",
             "recording_session_exceptions"}
    if unknown := set(data) - known:
        raise ValueError(f"{path}: unknown keys {', '.join(sorted(unknown))}")
    root = (path.parent / data.get("root", ".")).resolve()
    steps = data.get("steps", list(STEP_INPUTS))
    if unknown_steps := [step for step in steps if step not in STEP_INPUTS]:
        raise ValueError(f"{path}: unknown steps {', '.join(unknown_steps)} (known: {', '.join(STEP_INPUTS)})")
    steps = [step for step in STEP_INPUTS if step in steps]  # in dependency order
    step_options = data.get("step_options", {})
    for step, options in step_options.items():
        if step not in STEP_INPUTS:
            raise ValueError(f"{path}: step_options of unknown step {step}")
        _check_step_options(step, options, str(path))
    raw_defaults = data.get("defaults", {})
    defaults_version = raw_defaults.get("schema_version", SCHEMA_VERSION)
    defaults = _recording_session_part(raw_defaults, defaults_version, path.parent, f"{path} [defaults]")
    if "work_dir" not in data:
        raise ValueError(f"{path}: work_dir is required (outside the recording session folders), "
                         f"e.g. work_dir = \"derivatives/ephysalis/{RECORDING_SESSION_PLACEHOLDER}\"")
    conversion_batch = ConversionBatch(
        conversion_batch_file=path, conversion_batch_name=data.get("conversion_batch_name", path.parent.name),
        root=root, state_dir=(root / data.get("state_dir", ".ephysalis")).resolve(), steps=steps)
    session_root = (root / data.get("recording_session_root", ".")).resolve()
    folders = []
    for parent, names in data.get("recording_sessions", {}).items():
        if not isinstance(names, list):
            raise ValueError(f"{path}: [recording_sessions] {parent} must be a list of folder names")
        folders += [name if parent in (".", "") else f"{parent.strip('/')}/{name}" for name in names]
    if duplicates := sorted({f for f in folders if folders.count(f) > 1}):
        raise ValueError(f"{path}: recording session folders listed twice: {', '.join(duplicates)}")
    ids = [f.replace("/", "_") for f in folders]
    if same_id := sorted({i for i in ids if ids.count(i) > 1} - set(duplicates)):
        raise ValueError(f"{path}: different folders with the same recording session id: {', '.join(same_id)}")
    exceptions = data.get("recording_session_exceptions", {})
    if unlisted := sorted(set(exceptions) - set(folders)):
        raise ValueError(f"{path}: recording_session_exceptions of folders not in [recording_sessions]: "
                         f"{', '.join(unlisted)}")
    for folder in folders:
        recording_session_id = folder.replace("/", "_")
        exception = exceptions.get(folder, {})
        source = f"{path} session {folder}"
        if unknown := set(exception) - {"recording_session_config_file", "session_config", "step_options",
                                        "include"}:
            raise ValueError(f"{source}: unknown keys {', '.join(sorted(unknown))}")
        recording_session_path = (session_root / folder).resolve()
        merged, files = defaults, [path]
        config_template = exception.get("recording_session_config_file", data.get("recording_session_config_file"))
        if config_template:
            config_file = (root / _fill(config_template, recording_session_id, source)).resolve()
            if not config_file.is_file():
                raise ValueError(f"{source}: recording session config file {config_file} not found")
            raw = tomllib.loads(config_file.read_text(encoding="utf-8"))
            merged = merge_tables(merged, _recording_session_part(raw, raw.get("schema_version", SCHEMA_VERSION),
                                                                  config_file.parent, config_file))
            files.append(config_file)
        if "session_config" in exception:
            merged = merge_tables(merged, _recording_session_part(exception["session_config"], defaults_version,
                                                                  path.parent, f"{source} session_config"))
        merged.setdefault("session_id", recording_session_id)
        config = recording_session_config_from_dict(merged, source=source)
        for step, step_opts in exception.get("step_options", {}).items():
            if step not in STEP_INPUTS:
                raise ValueError(f"{source}: step_options of unknown step {step}")
            _check_step_options(step, step_opts, source)
        options = merge_tables(step_options, exception.get("step_options", {}))
        options = {step: {k: _fill(v, recording_session_id, source) for k, v in opts.items()}
                   for step, opts in options.items()}
        referenced = [Path(probe.geometry.file) for probe in config.probes.values()
                      if probe.geometry and probe.geometry.file]
        conversion_batch.recording_session_jobs.append(RecordingSessionJob(
            recording_session_id=recording_session_id,
            recording_session_path=recording_session_path,
            work_dir=(root / _fill(data["work_dir"], recording_session_id, source)).resolve(),
            recording_session_config=config, config_files=files, referenced_files=referenced, step_options=options,
            include=exception.get("include", True)))
    _check_outputs_outside_sources(conversion_batch)
    return conversion_batch


def _check_outputs_outside_sources(conversion_batch: ConversionBatch) -> None:
    """Work dirs, the state dir and the sort scratch dirs must not lie in a recording session folder (the
    source data): the steps write and delete there."""
    sources = [s.recording_session_path for s in conversion_batch.recording_session_jobs]
    for recording_session_job in conversion_batch.recording_session_jobs:
        scratch = recording_session_job.step_options.get("spike-sort", {}).get("scratch_dir")
        outputs = {"work_dir": recording_session_job.work_dir, "state_dir": conversion_batch.state_dir}
        if scratch:
            outputs["sort scratch_dir"] = (conversion_batch.root / scratch).resolve()
        for what, folder in outputs.items():
            for source in sources:
                if folder == source or folder.is_relative_to(source) or source.is_relative_to(folder):
                    raise ValueError(f"{conversion_batch.conversion_batch_file}: {what} {folder} of "
                                     f"{recording_session_job.recording_session_id} overlaps the recording "
                                     f"session folder {source} (source data are never written)")


def _fill(value: Any, recording_session_id: str, source: object = "") -> Any:
    """``value`` with the placeholders ``{recording_session}``, ``{subject}``, ``{session_label}`` filled in
    (also inside lists); ``{subject}`` / ``{session_label}`` need an id ``sub-<subject>_ses-<session_label>``."""
    if isinstance(value, list):
        return [_fill(v, recording_session_id, source) for v in value]
    if not isinstance(value, str):
        return value
    value = value.replace(RECORDING_SESSION_PLACEHOLDER, recording_session_id)
    if "{subject}" in value or "{session_label}" in value:
        parts = BIDS_RECORDING_SESSION_ID.match(recording_session_id)
        if not parts:
            raise ValueError(f"{source}: {{subject}} / {{session_label}} need a recording session id "
                             f"sub-<subject>_ses-<session_label>, not {recording_session_id!r}")
        value = value.replace("{subject}", parts["subject"]).replace("{session_label}", parts["session_label"])
    return value
