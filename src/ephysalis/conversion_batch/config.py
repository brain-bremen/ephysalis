"""``conversion_batch.toml``: many recording sessions processed alike (:mod:`ephysalis.conversion_batch.runner`).

The file describes the whole pipeline from the recording session folders to the outputs: the software, the
recording sessions, their configs, and every setting of every step. Example (paths are relative to ``root``, which
is relative to the file; default: the file's folder)::

    schema_version = 5
    conversion_batch_name = "V1 array recordings of monkeys B and F"
    root = ".."
    work_dir = "derivatives/ephysalis/{recording_session}"
    state_dir = "derivatives/ephysalis"            # pause file and status table (default: <root>/.ephysalis)
    recording_session_root = "sourcedata"          # the folders are relative to it (default: root)
    recording_session_config_file = "code/sessions/{recording_session}/session.toml"   # optional
    recording_sessions_file = "code/sessions/recording_sessions.toml"   # optional: a (generated) list
    recording_sessions = [                         # one recording session per line (after those of the file)
        { subject = "f", folder = "sub-f/ses-20091203", steps = { lfp = { rate_hz = 500.0 } } },
    ]

    [source_cache]                                 # how the steps read the session folders (default: direct)
    mode = "local"                                 # direct | memory (a copy on a RAM disk) | local (a local disk)
    dir = "/data/ephysalis-cache"                  # memory: default /dev/shm/ephysalis; local: required
    max_size_gb = 500                              # all copies together (optional)
    reserve_gb = 10                                # left free on the disk / in memory (default 10 / 32)

    [software]                                     # minimum versions; newer ones run the file too
    ephysalis = "0.3.0"
    ephysalis-bremen = "0.2.0"                     # every package of a plugin the sessions use

    [defaults]                                     # session.toml content of every recording session
    lab = "Kreiter lab"

    [subject_defaults.b]                           # session.toml content of the recording sessions of subject b
    input_format = "mcd"                           # the plugins: required (no detection in a batch)
    metadata_plugin = "vstim"
    experimenter = ["I. G."]
    [subject_defaults.b.aux_channels.digital]
    trial_start = { bit = 0, description = "trial start" }

    [steps.lfp]                                    # the steps that run; every setting is required
    probes = []
    rate_hz = 1000.0
    max_denominator = 10000
    anti_alias_kaiser_beta = 5.0
    anti_alias_half_length = 10
    anti_alias_cutoff = 1.0
    n_jobs = 4

    [steps.export]
    output_format = "dh5"
    session_label = "{session_label}"
    # ... (all settings of the step)

    [sessions."sub-b/ses-20101209"]                # decisions about single sessions of the lists
    include = false                                # e.g. recording broken after 10 min

A batch file can build on another one: ``extends = "pipeline.toml"`` (relative to the file) reads that file first;
the tables of this one are merged into it key by key, its values (also ``recording_sessions``) replace the base's
(``recording_sessions_file = ""``: none). E.g. a trial run on a few days with its own work dir and BIDS root.
Decisions in ``[sessions]`` of the base apply only to the sessions this file lists as well.

The recording sessions are those of ``recording_sessions_file`` (a TOML file with ``recording_sessions = [...]``,
e.g. generated from an inventory) and of ``recording_sessions``. Each entry: ``subject`` (the session's ``subject_id``) and ``folder`` (below
``recording_session_root``), optionally ``include`` (false: listed but not run), ``recording_session_config_file``
(instead of the template), ``session_config`` (session.toml content, after the file) and ``steps`` (settings of
this session that differ from ``[steps]``); ``[sessions."<folder>"]`` adds or replaces these optional keys of a
listed session (so that a generated list stays as it is). The id of a recording session is its folder with ``/`` as ``_``
(``sub-b_ses-20101207``); it names its work dir, its row in the status table and its ``--recording-sessions``
selection. Placeholders in ``work_dir``, ``recording_session_config_file`` and step settings:
``{recording_session}`` = the id, ``{subject}`` = the subject, ``{session_label}`` = the last folder name without
``ses-``.

The steps that run are the ``[steps.<step>]`` tables, in the order of the pipeline; each lists every setting of
its step (see the configuration reference: ``ephysalis.step_settings``). Every recording session's config names
its ``input_format`` and ``metadata_plugin``, and ``[software]`` lists ephysalis and the package of every plugin
used with the version the file was made for; ``ephysalis run`` refuses older installed versions. Newer versions
read the file: a format change comes with a migration (``CONVERSION_BATCH_MIGRATIONS``).

``[source_cache]`` (:mod:`ephysalis.conversion_batch.source_cache`): ``memory`` and ``local`` copy each session's
folder once before its first step that has to run and let the steps read the copy (deleted when the session
ends); the copies together stay below ``max_size_gb`` and leave ``reserve_gb`` free, a session waits for space and
one that alone is too big is read directly. It changes only where the data are read from (provenance records the
original folder), not the results; ``ephysalis run --source-cache`` overrides it for one run.

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
from ephysalis.conversion_batch.source_cache import SourceCacheSettings, source_cache_settings

CONVERSION_BATCH_SCHEMA_VERSION = 5


# CONVERSION_BATCH_MIGRATIONS[n]: raw data of version n -> n + 1 (versions before 3 are not read)
CONVERSION_BATCH_MIGRATIONS: dict[int, Any] = {
    3: lambda data: data,  # 4 added recording_sessions_file and [sessions] (optional)
    4: lambda data: data,  # 5 added [source_cache] (optional)
}
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


@dataclass
class RecordingSessionJob:
    recording_session_id: str  # {recording_session} in work_dir and in step settings
    recording_session_path: Path  # the recording's session folder
    work_dir: Path
    recording_session_config: RecordingSessionConfig  # defaults < subject defaults < session file < session_config
    config_files: list[Path]  # the files the config was made from (conversion batch file, override file)
    referenced_files: list[Path]  # files the config names (probe geometry)
    step_settings: dict[str, dict[str, Any]]  # step -> every setting of the step (ephysalis.step_settings)
    include: bool = True


@dataclass
class ConversionBatch:
    conversion_batch_file: Path  # conversion_batch.toml
    conversion_batch_name: str
    root: Path  # relative paths of the conversion batch file are relative to it
    state_dir: Path
    steps: list[str]  # in the order they run
    software: dict[str, str] = field(default_factory=dict)  # package -> minimum version
    source_cache: SourceCacheSettings = field(default_factory=SourceCacheSettings)  # [source_cache]
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


def check_software(software: dict[str, str], source: object) -> None:
    """Every package of ``[software]`` is installed with at least that version (``ephysalis`` must be listed)."""
    from importlib import metadata

    from packaging.version import Version

    if "ephysalis" not in software:
        raise ValueError(f"{source}: [software] must give the ephysalis version the file was made for")
    for package, required in software.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            raise ValueError(f"{source}: [software] {package} {required} is not installed") from None
        if Version(installed) < Version(required):
            raise ValueError(f"{source}: [software] needs {package} >= {required}, installed is {installed}")


def plugin_package(group: str, plugin_name: str) -> str | None:
    """Distribution (package) that provides the plugin ``plugin_name`` of the entry point ``group``."""
    from importlib import metadata

    from packaging.utils import canonicalize_name

    for entry_point in metadata.entry_points(group=group):
        if entry_point.name == plugin_name:
            return canonicalize_name(entry_point.dist.name) if entry_point.dist else None
    return None


def _check_plugins(config: RecordingSessionConfig, software: dict[str, str], source: object) -> None:
    """The session names its input format and metadata plugin, and their packages are in ``[software]``."""
    from packaging.utils import canonicalize_name

    from ephysalis.metadata.base import NO_METADATA_PLUGIN
    from ephysalis.plugins import INPUT_FORMAT_GROUP, METADATA_PLUGIN_GROUP

    if not config.input_format:
        raise ValueError(f"{source}: input_format is not set (a conversion batch names the input format plugin)")
    if not config.metadata_plugin:
        raise ValueError(f"{source}: metadata_plugin is not set (a conversion batch names the metadata plugin; "
                         f"'{NO_METADATA_PLUGIN}' for none)")
    listed = {canonicalize_name(p) for p in software}
    used = [(INPUT_FORMAT_GROUP, config.input_format), (METADATA_PLUGIN_GROUP, config.metadata_plugin),
            *((METADATA_PLUGIN_GROUP, r.metadata_plugin) for r in config.runs if r.metadata_plugin)]
    for group, name in used:
        if group == METADATA_PLUGIN_GROUP and name == NO_METADATA_PLUGIN:
            continue
        package = plugin_package(group, name)
        if package is None:
            raise ValueError(f"{source}: no {group} plugin {name!r} is installed")
        if package not in listed:
            raise ValueError(f"{source}: plugin {name!r} comes from {package}, which is not in [software]")


def _check_step_settings(step: str, settings: dict[str, Any], source: object, complete: bool) -> None:
    from ephysalis.step_settings import STEP_SETTINGS, missing_step_settings

    if step not in STEP_SETTINGS:
        raise ValueError(f"{source}: unknown step {step} (known: {', '.join(STEP_SETTINGS)})")
    if unknown := set(settings) - set(STEP_SETTINGS[step].model_fields):
        raise ValueError(f"{source}: unknown settings of {step}: {', '.join(sorted(unknown))} "
                         f"(known: {', '.join(STEP_SETTINGS[step].model_fields)})")
    if complete and (missing := missing_step_settings(step, settings)):
        raise ValueError(f"{source}: [steps.{step}] lacks {', '.join(missing)} (every setting is required)")


def _read_batch_file(path: Path, extended_by: tuple[Path, ...]) -> tuple[dict[str, Any], set[str]]:
    """The raw data of a batch file, migrated, on top of the file it ``extends`` (tables merged key by key, values
    replaced); and the folders of ``[sessions]`` that come from the base only (they apply only to the sessions
    this file lists as well)."""
    data = migrate(tomllib.loads(path.read_text(encoding="utf-8")), CONVERSION_BATCH_SCHEMA_VERSION,
                   CONVERSION_BATCH_MIGRATIONS, missing=CONVERSION_BATCH_SCHEMA_VERSION, source=path)
    if "extends" not in data:
        return data, set()
    base_path = (path.parent / data.pop("extends")).resolve()
    if base_path == path or base_path in extended_by:
        raise ValueError(f"{path}: extends {base_path}, which extends it again")
    base, base_inherited = _read_batch_file(base_path, (*extended_by, path))
    own = set(data.get("sessions", {}))
    return merge_tables(base, data), (set(base.get("sessions", {})) | base_inherited) - own


def load_conversion_batch(path: str | Path, check_installed_software: bool = True) -> ConversionBatch:
    """The conversion batch of ``path``; ``check_installed_software``: refuse older installed versions than
    ``[software]`` says (off only to look at a file)."""
    path = Path(path).resolve()
    data, inherited_sessions = _read_batch_file(path, ())
    known = {"schema_version", "conversion_batch_name", "root", "work_dir", "state_dir", "software", "defaults",
             "subject_defaults", "steps", "recording_session_root", "recording_session_config_file",
             "recording_sessions_file", "recording_sessions", "sessions", "source_cache"}
    if unknown := set(data) - known:
        raise ValueError(f"{path}: unknown keys {', '.join(sorted(unknown))}")
    root = (path.parent / data.get("root", ".")).resolve()
    software = {str(k): str(v) for k, v in data.get("software", {}).items()}
    if check_installed_software:
        check_software(software, path)
    step_settings = data.get("steps", {})
    for step, settings in step_settings.items():
        _check_step_settings(step, settings, path, complete=True)
    steps = [step for step in STEP_INPUTS if step in step_settings]  # in the order they run
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
        root=root, state_dir=(root / data.get("state_dir", ".ephysalis")).resolve(), steps=steps, software=software,
        source_cache=source_cache_settings(data.get("source_cache", {}), root, path))
    session_root = (root / data.get("recording_session_root", ".")).resolve()
    entries = []
    if data.get("recording_sessions_file"):  # "" in a file that extends another: none
        sessions_file = (root / data["recording_sessions_file"]).resolve()
        listed = tomllib.loads(sessions_file.read_text(encoding="utf-8"))
        if set(listed) - {"recording_sessions"}:
            raise ValueError(f"{sessions_file}: only recording_sessions = [...] is read from it")
        entries += listed.get("recording_sessions", [])
    entries += data.get("recording_sessions", [])
    if not all(isinstance(e, dict) for e in entries):
        raise ValueError(f"{path}: recording_sessions must be a list of tables {{ subject = ..., folder = ... }}")
    listed_folders = {str(e.get("folder", "")).strip("/") for e in entries}
    decisions = {f: d for f, d in data.get("sessions", {}).items() if f in listed_folders or f not in inherited_sessions}
    if unlisted := sorted(set(decisions) - listed_folders):
        raise ValueError(f"{path}: [sessions] of folders that are not listed: {', '.join(unlisted)}")
    for folder, decision in decisions.items():
        if bad := set(decision) - {"include", "recording_session_config_file", "session_config", "steps"}:
            raise ValueError(f"{path}: [sessions.\"{folder}\"] can only set include, recording_session_config_file, "
                             f"session_config, steps (not {', '.join(sorted(bad))})")
    entries = [{**e, **decisions.get(str(e.get("folder", "")).strip("/"), {})} for e in entries]
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
                                    "session_config", "steps"}:
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
        _check_plugins(config, software, source)
        for step, settings in entry.get("steps", {}).items():
            if step not in step_settings:
                raise ValueError(f"{source}: settings of step {step}, which the batch does not run ([steps.{step}])")
            _check_step_settings(step, settings, source, complete=False)
        session_settings = {step: {k: _fill(v, placeholders) for k, v in
                                   merge_tables(settings, entry.get("steps", {}).get(step, {})).items()}
                            for step, settings in step_settings.items()}
        from ephysalis.step_settings import STEP_SETTINGS

        for step, settings in session_settings.items():
            try:
                STEP_SETTINGS[step].model_validate(settings)
            except Exception as error:  # noqa: BLE001 (pydantic's message names the setting)
                raise ValueError(f"{source}: [steps.{step}]: {error}") from None
        referenced = [Path(probe.geometry.file) for probe in config.probes.values()
                      if probe.geometry and probe.geometry.file]
        conversion_batch.recording_session_jobs.append(RecordingSessionJob(
            recording_session_id=recording_session_id,
            recording_session_path=recording_session_path,
            work_dir=(root / _fill(data["work_dir"], placeholders)).resolve(),
            recording_session_config=config, config_files=files, referenced_files=referenced,
            step_settings=session_settings,
            include=entry.get("include", True)))
    _check_outputs_outside_sources(conversion_batch)
    _check_source_cache_outside(conversion_batch, [session_root / folder for folder in folders])
    return conversion_batch


def _check_source_cache_outside(conversion_batch: ConversionBatch, listed_folders: list[Path]) -> None:
    """The source cache dir must not overlap a session folder (also as listed, before symbolic links are
    followed) or a work dir: its copies are deleted."""
    folder = conversion_batch.source_cache.dir
    if conversion_batch.source_cache.mode == "direct" or folder is None:
        return
    folder = folder.resolve()
    for job, listed in zip(conversion_batch.recording_session_jobs, listed_folders):
        for what, other in (("recording session folder", job.recording_session_path),
                            ("recording session folder", listed), ("work dir", job.work_dir)):
            if folder == other or folder.is_relative_to(other) or other.is_relative_to(folder):
                raise ValueError(f"{conversion_batch.conversion_batch_file}: [source_cache] dir {folder} overlaps the "
                                 f"{what} {other} of {job.recording_session_id}")


def _check_outputs_outside_sources(conversion_batch: ConversionBatch) -> None:
    """Work dirs, the state dir and the sort scratch dirs must not lie in a recording session folder (the
    source data): the steps write and delete there."""
    sources = [s.recording_session_path for s in conversion_batch.recording_session_jobs]
    for recording_session_job in conversion_batch.recording_session_jobs:
        scratch = recording_session_job.step_settings.get("spike-sort", {}).get("scratch_dir")
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
