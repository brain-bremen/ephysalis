"""``conversion_batch.toml``: many recording sessions processed alike (:mod:`ephysalis.conversion_batch.runner`).

Example (paths are relative to ``root``, which is relative to the file; default: the file's folder)::

    schema_version = 2
    conversion_batch_name = "V1 array recordings of monkeys B and F"
    root = ".."
    work_dir = "derivatives/ephysalis/{recording_session}"
    state_dir = "derivatives/ephysalis"            # pause file and status table (default: <root>/.ephysalis)
    steps = ["align-clocks", "extract-events", "lfp", "export"]
    recording_session_root = "sourcedata"          # the folders are relative to it (default: root)
    recording_session_config_file = "code/sessions/{recording_session}/session.toml"   # optional
    recording_sessions = [                         # one recording session per line
        { subject = "b", folder = "sub-b/ses-20101207" },
        { subject = "b", folder = "sub-b/ses-20101209", include = false },
        { subject = "f", folder = "sub-f/ses-20091203", recording_session_config_file = "code/f_20091203.toml" },
    ]

    [defaults]                                     # session.toml content of every recording session
    schema_version = 10
    lab = "Kreiter lab"

    [subject_defaults.b]                           # session.toml content of the recording sessions of subject b
    experimenter = ["I. G."]
    [subject_defaults.b.aux_channels.digital]
    trial_start = { bit = 0, description = "trial start" }

    [step_options.lfp]                             # options of a step, as on its command line
    n_jobs = 4
    [step_options.export]
    output_format = "dh5"
    session_label = "{session_label}"

Each entry of ``recording_sessions``: ``subject`` (the session's ``subject_id``) and ``folder`` (below
``recording_session_root``), optionally ``include`` (false: listed but not run), ``recording_session_config_file``
(instead of the template), ``session_config`` (session.toml content, after the file) and ``step_options`` (this
session's step options). The id of a recording session is its folder with ``/`` as ``_`` (``sub-b_ses-20101207``);
it names its work dir, its row in the status table and its ``--recording-sessions`` selection. Placeholders in
``work_dir``, ``recording_session_config_file`` and step options: ``{recording_session}`` = the id, ``{subject}`` =
the subject, ``{session_label}`` = the last folder name without ``ses-``. The input format of each recording
session is detected from its files (the input format plugins), or set by ``input_format`` in its config.

A recording session's config is ``[defaults]``, overridden by ``[subject_defaults.<subject>]``, by its
``recording_session_config_file`` (if the template is given, the file must exist) and by its ``session_config``:
tables are merged key by key, everything else (values, arrays such as ``runs``) is replaced. Every part must have
the current ``schema_version`` (or none). Relative probe geometry files are relative to the file that names them
(the defaults and ``session_config``: the conversion batch file).
"""

from __future__ import annotations

import argparse
import copy
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
             "subject_defaults"}
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
    for table, part in [("defaults", raw_defaults), *((f"subject_defaults.{k}", v) for k, v in
                                                      data.get("subject_defaults", {}).items())]:
        if "subject_id" in part:
            raise ValueError(f"{path}: [{table}] sets subject_id; it is the subject of each recording session")
    defaults_version = raw_defaults.get("schema_version", SCHEMA_VERSION)
    defaults = _recording_session_part(raw_defaults, defaults_version, path.parent, f"{path} [defaults]")
    if "work_dir" not in data:
        raise ValueError(f"{path}: work_dir is required (outside the recording session folders), "
                         f"e.g. work_dir = \"derivatives/ephysalis/{RECORDING_SESSION_PLACEHOLDER}\"")
    conversion_batch = ConversionBatch(
        conversion_batch_file=path, conversion_batch_name=data.get("conversion_batch_name", path.parent.name),
        root=root, state_dir=(root / data.get("state_dir", ".ephysalis")).resolve(), steps=steps)
    session_root = (root / data.get("recording_session_root", ".")).resolve()
    entries = data.get("recording_sessions", [])
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError(f"{path}: recording_sessions must be a list of tables {{ subject = ..., folder = ... }}")
    subject_defaults = {subject: _recording_session_part(part, part.get("schema_version", defaults_version),
                                                         path.parent, f"{path} [subject_defaults.{subject}]")
                        for subject, part in data.get("subject_defaults", {}).items()}
    if unused := sorted(set(subject_defaults) - {e.get("subject") for e in entries}):
        raise ValueError(f"{path}: subject_defaults of subjects without recording sessions: {', '.join(unused)}")
    folders = [str(e.get("folder", "")).strip("/") for e in entries]
    ids = [f.replace("/", "_") for f in folders]
    if same := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"{path}: recording sessions listed twice (or folders with the same id): {', '.join(same)}")
    for entry, folder, recording_session_id in zip(entries, folders, ids):
        source = f"{path} session {folder or entry}"
        if not entry.get("subject") or not folder:
            raise ValueError(f"{source}: every recording session needs subject and folder")
        if unknown := set(entry) - {"subject", "folder", "include", "recording_session_config_file",
                                    "session_config", "step_options"}:
            raise ValueError(f"{source}: unknown keys {', '.join(sorted(unknown))}")
        subject = str(entry["subject"])
        session_label = folder.rsplit("/", 1)[-1].removeprefix("ses-")
        placeholders = {"{recording_session}": recording_session_id, "{subject}": subject,
                        "{session_label}": session_label}
        recording_session_path = (session_root / folder).resolve()
        merged, files = merge_tables(defaults, subject_defaults.get(subject, {})), [path]
        config_template = entry.get("recording_session_config_file", data.get("recording_session_config_file"))
        if config_template:
            config_file = (root / _fill(config_template, placeholders)).resolve()
            if not config_file.is_file():
                raise ValueError(f"{source}: recording session config file {config_file} not found")
            raw = tomllib.loads(config_file.read_text(encoding="utf-8"))
            merged = merge_tables(merged, _recording_session_part(raw, raw.get("schema_version", SCHEMA_VERSION),
                                                                  config_file.parent, config_file))
            files.append(config_file)
        if "session_config" in entry:
            merged = merge_tables(merged, _recording_session_part(entry["session_config"], defaults_version,
                                                                  path.parent, f"{source} session_config"))
        merged.setdefault("session_id", recording_session_id)
        if merged.get("subject_id", subject) != subject:
            raise ValueError(f"{source}: subject_id {merged['subject_id']!r} of its config is not the subject {subject!r}")
        merged["subject_id"] = subject
        config = recording_session_config_from_dict(merged, source=source)
        for step, step_opts in entry.get("step_options", {}).items():
            if step not in STEP_INPUTS:
                raise ValueError(f"{source}: step_options of unknown step {step}")
            _check_step_options(step, step_opts, source)
        options = merge_tables(step_options, entry.get("step_options", {}))
        options = {step: {k: _fill(v, placeholders) for k, v in opts.items()}
                   for step, opts in options.items()}
        referenced = [Path(probe.geometry.file) for probe in config.probes.values()
                      if probe.geometry and probe.geometry.file]
        conversion_batch.recording_session_jobs.append(RecordingSessionJob(
            recording_session_id=recording_session_id,
            recording_session_path=recording_session_path,
            work_dir=(root / _fill(data["work_dir"], placeholders)).resolve(),
            recording_session_config=config, config_files=files, referenced_files=referenced, step_options=options,
            include=entry.get("include", True)))
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


def _fill(value: Any, placeholders: dict[str, str]) -> Any:
    """``value`` with the placeholders (``{recording_session}``, ``{subject}``, ``{session_label}``) filled in,
    also inside lists."""
    if isinstance(value, list):
        return [_fill(v, placeholders) for v in value]
    if not isinstance(value, str):
        return value
    for placeholder, text in placeholders.items():
        value = value.replace(placeholder, text)
    return value
