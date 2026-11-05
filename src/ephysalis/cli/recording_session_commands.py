"""Pipeline commands: ``ephysalis inspect``, ``init``, ``sync``, ``timing``, ``lfp``, ``mua``, ``sort``,
``export`` and ``reproduce``.

Every step takes one or more sessions. With several sessions, per-session paths contain the
placeholder ``{session}`` (the session folder name), e.g. ``--work-dir D:/work/{session}``;
without ``--work-dir`` each session uses ``<session>/ephysalis``. A failing session does not
stop the others; the exit code is 1 if any session failed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from ephysalis.recording_session.model import Run, RecordingSession
from ephysalis.recording_session.config import (
    RECORDING_SESSION_CONFIG_FILENAME,
    RecordingSessionConfig,
    default_recording_session_config,
    load_recording_session_config,
    save_recording_session_config,
)
from ephysalis.formats.input.base import open_recording_session
from ephysalis.recording_session.probes import configure_probes
from ephysalis.plugins import ComputesChannelGainCalibration, HasSynchronisedTimestamps
from ephysalis.timeline.clocks import RunSync, SyncEdgeCache

from ephysalis.step_settings import add_step_settings_arguments, step_settings_from_args  # noqa: E402

logger = logging.getLogger(__name__)
SESSION_PLACEHOLDER = "{session}"
# options that hold a path of one session: need SESSION_PLACEHOLDER with several sessions
_PER_RECORDING_SESSION_OPTIONS = ("work_dir", "config", "cache_dir", "output", "existing")


def per_recording_session(func: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
    """Run a one-session command for every session in ``args.sessions``."""

    def run(args: argparse.Namespace) -> int:
        recording_sessions = [Path(s) for s in args.recording_sessions]
        if len(recording_sessions) > 1:
            fixed = [f"--{o.replace('_', '-')}" for o in _PER_RECORDING_SESSION_OPTIONS
                     if getattr(args, o, None) and SESSION_PLACEHOLDER not in str(getattr(args, o))]
            if fixed:
                print(f"{', '.join(fixed)} with several sessions needs the placeholder {SESSION_PLACEHOLDER}, "
                      f"e.g. --work-dir D:/work/{SESSION_PLACEHOLDER}", file=sys.stderr)
                return 2
        failed = []
        for recording_session in recording_sessions:
            one = argparse.Namespace(**vars(args))
            one.recording_session = recording_session
            for option in _PER_RECORDING_SESSION_OPTIONS:
                value = getattr(one, option, None)
                if value is not None:
                    setattr(one, option, str(value).replace(SESSION_PLACEHOLDER, recording_session.name))
            if len(recording_sessions) > 1:
                print(f"=== {recording_session.name} ===", flush=True)
            try:
                code = func(one)
            except KeyboardInterrupt:
                raise
            except Exception:
                if len(recording_sessions) == 1:
                    raise
                logger.exception("%s failed", recording_session.name)
                code = 1
            if code:
                failed.append(recording_session.name)
        if len(recording_sessions) > 1:
            print(f"{len(recording_sessions) - len(failed)} of {len(recording_sessions)} sessions done"
                  + (f"; failed: {', '.join(failed)}" if failed else ""))
        return 1 if failed else 0

    return run


def _inspect(args: argparse.Namespace) -> int:
    recording_session = open_recording_session(args.recording_session, args.format)
    config_path = recording_session.path / RECORDING_SESSION_CONFIG_FILENAME
    config = load_recording_session_config(config_path) if config_path.exists() else None
    configure_probes(recording_session, config, config_path.parent)

    print(f"Session {recording_session.name} ({recording_session.format.input_format_name}), {len(recording_session.runs)} runs")
    print(f"Config: {config_path if config else 'none (run `ephysalis init`)'}")
    for run in recording_session.runs:
        run_config = config.run_config(run.name) if config else None
        flag = "" if run_config is None or run_config.include else "  [excluded]"
        start = run.start_time.isoformat(sep=" ") if run.start_time else "?"
        duration = f"{run.duration_s:.0f} s" if run.duration_s else "?"
        print(f"\n  [{run.index}] {run.name}  start {start}  {duration}{flag}")
        for probe in run.probes.values():
            bands = "+".join(k.value for k in probe.streams)
            print(
                f"      {probe.probe_id}: {probe.part_number or '?'} sn {probe.serial_number or '?'}, "
                f"{probe.num_channels} ch, {probe.num_shanks} shank(s), streams {bands}"
            )
        if run.aux_stream:
            print(f"      aux: {run.aux_stream} {run.extra.get('aux', '')}")
        others = [s for s in run.streams if s != run.aux_stream
                  and all(s not in info.streams.values() for info in run.probes.values())]
        if others:  # e.g. for probes defined in session.toml
            print(f"      other streams: {', '.join(others)}")

        from ephysalis.metadata.base import run_trial_tables

        plugin, tables = run_trial_tables(run, None, None, config)  # without times: no clock alignment
        if plugin is None:
            print("      trials: no metadata plugin recognises the run")
        for experimental_protocol, table in tables.items():
            line = f"      trials [{plugin.metadata_plugin_name}] {experimental_protocol}: {len(table)} trials"
            if args.conditions:
                line += f", {table['condition'].nunique()} conditions"
            print(line)
        if args.timing and (run_config is None or run_config.include):
            _print_timing(run, config, _sync_cache(recording_session.path, args.cache_dir))
    return 0


def _sync_cache(recording_session_path: Path, cache_dir: str | None) -> SyncEdgeCache:
    from ephysalis.timeline.steps import default_sync_cache

    return SyncEdgeCache(cache_dir) if cache_dir else default_sync_cache(recording_session_path)


def _print_timing(run: Run, config: RecordingSessionConfig | None, cache: SyncEdgeCache) -> None:
    from ephysalis.timeline.steps import extract_run_timing

    timing = extract_run_timing(run, config, cache)
    for stream, clock in timing.sync.maps.items():
        if stream == timing.sync.reference:
            continue
        print(
            f"      sync {stream} -> {timing.sync.reference}: {clock.source_samples.size} edges, "
            f"{clock.effective_sampling_frequency:.4f} Hz, max residual {clock.max_residual_s() * 1e3:.3f} ms"
        )
    for name, trials in timing.trials.items():
        first = trials["start_time"].iloc[0] if len(trials) else float("nan")
        print(
            f"      trials {name}: {len(trials)} detected ({trials.attrs.get('photodiode_mode', '?')} "
            f"photodiode), first at {first:.3f} s"
        )
    if timing.running_speed is not None:
        print(f"      running speed: max {timing.running_speed.max():.1f} cm/s")


def _init(args: argparse.Namespace) -> int:
    recording_session = open_recording_session(args.recording_session, args.format)
    output = Path(args.output) if args.output else recording_session.path / RECORDING_SESSION_CONFIG_FILENAME
    if output.exists() and not args.force:
        print(f"{output} exists, use --force to overwrite", file=sys.stderr)
        return 1

    from ephysalis.plugins import AddsRecordingSessionConfigFromMetadataFiles, metadata_plugins

    config, warnings = default_recording_session_config(recording_session), []  # what the recording says
    metadata_file = Path(args.metadata) if args.metadata else None
    for plugin in metadata_plugins():  # what the setup's metadata files say
        if isinstance(plugin, AddsRecordingSessionConfigFromMetadataFiles):
            warnings += plugin.add_recording_session_config_from_metadata_files(recording_session, config, metadata_file)

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    save_recording_session_config(config, output)
    print(f"Wrote {output}")
    return 0


def _align_clocks(args: argparse.Namespace) -> int:
    from ephysalis.timeline.clocks import load_or_extract_edges, synchronize_run
    from ephysalis.timeline.steps import aux_channel_map, default_work_dir, write_clock_report

    recording_session, work_dir, config, config_path = _open_with_config(args)
    _configure(recording_session, config, config_path, work_dir)
    settings = step_settings_from_args("align-clocks", args)
    cache = SyncEdgeCache(settings.cache_dir) if settings.cache_dir else SyncEdgeCache(work_dir / "sync")
    for run in recording_session.runs:
        run_config = config.run_config(run.name) if config else None
        if args.runs and run.name not in args.runs:
            continue
        if not args.runs and run_config is not None and not run_config.include:
            continue
        if args.force and cache.path(run.name).exists():
            cache.path(run.name).unlink()
        if not (isinstance(run.format, HasSynchronisedTimestamps)  # formats with synchronised timestamps have no edges
                and run.format.clock_alignment_from_synchronised_timestamps(run, None) is not None):
            edges = load_or_extract_edges(run, aux_channel_map(run, config), cache)
            counts = ", ".join(f"{stream} {e.samples.size}" for stream, e in edges.items())
            print(f"{run.name}: sync edges {counts}")
        sync = synchronize_run(run, aux_channel_map(run, config), cache=cache)
        write_clock_report(work_dir, run.name, sync)
        for stream, clock in sync.maps.items():
            if stream != sync.reference:
                print(f"{run.name}: {stream} -> {sync.reference} {clock.effective_sampling_frequency:.4f} Hz, "
                      f"max residual {clock.max_residual_s() * 1e3:.3f} ms")
    _record_step("align-clocks", args, recording_session, work_dir, config, config_path)
    print(f"Sync edge cache: {cache.directory}, clock alignment: {work_dir / 'clocks'}")
    return 0


def _record_step(step: str, args: argparse.Namespace, recording_session: RecordingSession, work_dir: Path,
                 config: RecordingSessionConfig | None, config_path: Path | None) -> dict:
    """Append this invocation to ``<work dir>/provenance/<step>.json``; returns its record."""
    from ephysalis.provenance import step_record, write_step_record

    options = {k: v for k, v in vars(args).items()
               if k not in ("func", "command", "recording_session", "recording_sessions", "argv", "verbose") and not callable(v)}
    record = step_record(step, recording_session, config, config_path, options, getattr(args, "argv", None), work_dir)
    write_step_record(work_dir, record)
    return record


def _parse_value(text: str) -> object:
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _included_run_syncs(recording_session: RecordingSession, config: RecordingSessionConfig | None, work_dir: Path) -> dict[str, RunSync]:
    """RunSync of every included run, from the sync edge cache in the work dir."""
    from ephysalis.timeline.clocks import synchronize_run
    from ephysalis.timeline.steps import aux_channel_map

    cache = SyncEdgeCache(work_dir / "sync")
    syncs = {}
    for run in recording_session.runs:
        run_config = config.run_config(run.name) if config else None
        if run_config is None or run_config.include:
            syncs[run.name] = synchronize_run(run, aux_channel_map(run, config), cache=cache)
    return syncs


def _spike_sort(args: argparse.Namespace) -> int:
    import tempfile

    from ephysalis.spike_sorting.spikes import spike_sort_probe
    from ephysalis.timeline.steps import default_work_dir

    recording_session, work_dir, config, config_path = _open_with_config(args)
    _configure(recording_session, config, config_path, work_dir)
    print(f"Config: {config_path or 'none (sorting all channels)'}")
    syncs = _included_run_syncs(recording_session, config, work_dir)
    settings = step_settings_from_args("spike-sort", args)
    params = dict(kv.split("=", 1) for kv in settings.sorter_params)
    scratch = Path(settings.scratch_dir) if settings.scratch_dir else Path(tempfile.gettempdir()) / "ephysalis"
    for probe_id in settings.probes or recording_session.probe_ids:
        runs = spike_sort_probe(
            recording_session, probe_id, work_dir, scratch, config, syncs,
            sorter_params={k: _parse_value(v) for k, v in params.items()},
            n_jobs=settings.n_jobs, keep_binary=settings.keep_binary, sorter=settings.spike_sorter or None,
        )
        for run_name, spikes in runs.items():
            print(f"{probe_id} {run_name}: {len(spikes)} spikes, {spikes.unit_id.nunique()} units")
    _record_step("spike-sort", args, recording_session, work_dir, config, config_path)
    print(f"Output: {work_dir}")
    return 0


def _configure(recording_session: RecordingSession, config: RecordingSessionConfig | None, config_path: Path | None, work_dir: Path,
               calibration: bool = True, run_starts: bool = True) -> None:
    """Probes from session.toml, the calibration of the input format (``[calibration]``, ``ephysalis
    calibrate``) and the better run start times from the metadata plugin (``ephysalis extract-trials``,
    :mod:`ephysalis.metadata.runstart`). The extract steps take neither: they work on the recorded start
    times, so their outputs do not depend on what an earlier run left in the work dir."""
    from ephysalis.metadata.runstart import apply_run_starts

    config_dir = config_path.parent if config_path else None
    configure_probes(recording_session, config, config_dir)
    if calibration and isinstance(recording_session.format, ComputesChannelGainCalibration):  # not for steps without gains
        recording_session.format.apply_channel_gain_calibration(recording_session, config, config_dir, work_dir)
    if run_starts:
        apply_run_starts(recording_session, work_dir)


def _calibrate(args: argparse.Namespace) -> int:
    from ephysalis.timeline.steps import default_work_dir

    recording_session, work_dir, config, config_path = _open_with_config(args)
    if config is None or config.calibration.run is None:
        print("Nothing to do: [calibration] run is not set in session.toml")
        return 0
    configure_probes(recording_session, config, config_path.parent if config_path else None)
    calibration = (recording_session.format.compute_channel_gain_calibration(recording_session, config, work_dir)
                   if isinstance(recording_session.format, ComputesChannelGainCalibration) else None)
    if calibration is None:
        print(f"Nothing to do: the {recording_session.format.input_format_name} format has no calibration")
        return 0
    for probe_id, entry in calibration["probes"].items():
        gains = [g for g in entry["gains_uv"] if g is not None]
        print(f"{probe_id}: sine {entry['frequency_hz']:.3f} Hz in {entry['run']}, {len(gains)} channels calibrated "
              f"(median {sorted(gains)[len(gains) // 2] if gains else float('nan'):.4f} uV per count), "
              f"without signal: {', '.join(entry['without_signal']) or 'none'}")
    _record_step("calibrate", args, recording_session, work_dir, config, config_path)
    print(f"Output: {work_dir}")
    return 0


def _quality(args: argparse.Namespace) -> int:
    from ephysalis.processing.quality import GOOD, assess_run
    from ephysalis.timeline.steps import default_work_dir

    recording_session, work_dir, config, config_path = _open_with_config(args)
    _configure(recording_session, config, config_path, work_dir)
    probes = {pid: p.quality for pid, p in (config.probes.items() if config else []) if p.quality is not None}
    if not probes:
        print("Nothing to do: no probe has [probes.<id>.quality] in session.toml")
        return 0
    for run in recording_session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        for probe_id, quality_config in probes.items():
            if probe_id in run.probes and (not args.probes or probe_id in args.probes):
                results = assess_run(run, probe_id, work_dir, quality_config)
                bad = [r["channel"] for r in results if r["status"] != GOOD]
                print(f"{run.name} {probe_id}: {len(results) - len(bad)} of {len(results)} channels good"
                      + (f", bad: {', '.join(bad)}" if bad else ""))
    _record_step("quality", args, recording_session, work_dir, config, config_path)
    print(f"Output: {work_dir / 'quality'}")
    return 0


def _csd(args: argparse.Namespace) -> int:

    from ephysalis.processing.csd import TooFewCsdInputs, csd_folder, extract_csd
    from ephysalis.timeline.steps import default_work_dir

    recording_session, work_dir, config, config_path = _open_with_config(args)
    _configure(recording_session, config, config_path, work_dir)
    probes = {pid: p.csd for pid, p in (config.probes.items() if config else []) if p.csd is not None}
    if not probes:
        print("Nothing to do: no probe has [probes.<id>.csd] in session.toml")
        return 0
    for run in recording_session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        for probe_id, csd_config in probes.items():
            if probe_id in run.probes and (not args.probes or probe_id in args.probes):
                try:
                    csd = extract_csd(run, probe_id, work_dir, csd_config, args.n_jobs, args.force)
                except TooFewCsdInputs as err:  # e.g. the whole array flagged bad in this run: no CSD
                    from ephysalis.filesystem import remove_folder

                    remove_folder(csd_folder(work_dir, run.name, probe_id))
                    logger.warning("%s %s: no CSD, %s", run.name, probe_id, err)
                    print(f"{run.name} {probe_id}: no CSD ({err})")
                    continue
                print(f"{run.name} {probe_id}: CSD {csd.get_num_channels()} channels, {csd.get_total_duration():.0f} s")
    _record_step("csd", args, recording_session, work_dir, config, config_path)
    print(f"Output: {work_dir / 'csd'}")
    return 0


def _open_with_config(args: argparse.Namespace) -> tuple[RecordingSession, Path, RecordingSessionConfig | None, Path | None]:
    """The recording session (in ``--format``, else its config's ``input_format``, else detected), its work dir and
    its config."""
    from ephysalis.timeline.steps import default_work_dir

    path = Path(args.recording_session)
    work_dir = Path(args.work_dir) if args.work_dir else default_work_dir(path)
    config, config_path = _load_config_for(path, work_dir, args.config)
    input_format = args.format or (config.input_format if config is not None else None)
    recording_session = open_recording_session(path, input_format)
    check_config_runs(recording_session, config, config_path)
    return recording_session, work_dir, config, config_path


def check_config_runs(recording_session: RecordingSession, config: RecordingSessionConfig | None, config_path: Path | None) -> None:
    """Stop when the config names runs the recording does not have: their settings (``include = false``, a task)
    would not apply, and the runs they meant would be processed with the defaults."""
    if config is None:
        return
    names = [run.name for run in recording_session.runs]
    if unknown := [run.name for run in config.runs if run.name not in names]:
        raise SystemExit(f"{config_path}: runs not in the recording {recording_session.path}: {', '.join(unknown)} "
                         f"(its runs: {', '.join(names)})")


def _load_config_for(recording_session_path: Path, work_dir: Path, config: str | None) -> tuple[RecordingSessionConfig | None, Path | None]:
    candidates = [Path(config)] if config else [recording_session_path / RECORDING_SESSION_CONFIG_FILENAME, work_dir / RECORDING_SESSION_CONFIG_FILENAME]
    path = next((p for p in candidates if p.exists() or config), None)
    return (load_recording_session_config(path) if path else None), path


def _extract_step(args: argparse.Namespace, step: str, extract: Callable[[Run, RunSync, RecordingSessionConfig | None, Path], None]
                  ) -> int:
    """Run ``extract(run, sync, config, work_dir)`` for every selected run on its clock alignment
    (from the sync edge cache, extracted on demand) and record ``step``."""
    from ephysalis.timeline.steps import align_run_clocks, default_work_dir

    recording_session, work_dir, config, config_path = _open_with_config(args)
    _configure(recording_session, config, config_path, work_dir, calibration=False, run_starts=False)
    cache = SyncEdgeCache(work_dir / "sync")
    for run in recording_session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        extract(run, align_run_clocks(run, config, cache), config, work_dir)
    _record_step(step, args, recording_session, work_dir, config, config_path)
    print(f"Output: {work_dir}")
    return 0


def _write_trials(run: Run, sync: RunSync, config: RecordingSessionConfig | None, work_dir: Path) -> None:
    from ephysalis.metadata.runstart import write_run_start
    from ephysalis.timeline.steps import extract_run_trials

    (work_dir / "trials").mkdir(parents=True, exist_ok=True)
    from ephysalis.filesystem import glob_in_work_folder

    for old in glob_in_work_folder(work_dir / "trials", f"{run.name}_*.csv"):
        old.unlink()
    trials, run_start = extract_run_trials(run, sync, config)
    for experimental_protocol, table in trials.items():
        if table.columns.empty:  # no trials matched (e.g. no trial of the .tdr inside the run)
            print(f"{run.name} {experimental_protocol}: no trials")
            continue
        table.to_csv(work_dir / "trials" / f"{run.name}_{experimental_protocol}.csv", index=False)
        print(f"{run.name} {experimental_protocol}: {len(table)} trials")
    write_run_start(work_dir, run.name, run.recorded_start_time or run.start_time, run_start)
    if run_start is not None:
        print(f"{run.name}: start {run_start.start_time.isoformat()} ({run_start.method})")


def _write_events(run: Run, sync: RunSync, config: RecordingSessionConfig | None, work_dir: Path) -> None:
    from ephysalis.timeline.steps import extract_run_events

    (work_dir / "events").mkdir(parents=True, exist_ok=True)
    events = extract_run_events(run, sync, config)
    events.to_csv(work_dir / "events" / f"{run.name}.csv", index=False)
    print(f"{run.name}: {len(events)} events")


def _write_behavior(run: Run, sync: RunSync, config: RecordingSessionConfig | None, work_dir: Path) -> None:
    import numpy as np

    from ephysalis.timeline.steps import extract_run_behavior

    running_speed, eye_position = extract_run_behavior(run, sync, config)
    for folder, signal, what in (("running_speed", running_speed, "running speed"), ("eye", eye_position, "eye position")):
        path = work_dir / folder / f"{run.name}.npy"
        path.unlink(missing_ok=True)
        if signal is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, signal.astype(np.float32))
            print(f"{run.name}: {what}, {len(signal)} samples")


def _extract_trials(args: argparse.Namespace) -> int:
    return _extract_step(args, "extract-trials", _write_trials)


def _extract_events(args: argparse.Namespace) -> int:
    return _extract_step(args, "extract-events", _write_events)


def _extract_behavior(args: argparse.Namespace) -> int:
    return _extract_step(args, "extract-behavior", _write_behavior)


def _lfp(args: argparse.Namespace) -> int:
    from ephysalis.processing.lfp import extract_lfp
    from ephysalis.timeline.steps import default_work_dir

    recording_session, work_dir, config, config_path = _open_with_config(args)
    _configure(recording_session, config, config_path, work_dir)
    settings = step_settings_from_args("lfp", args)
    for run in recording_session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        for probe_id in settings.probes or list(run.probes):
            lfp = extract_lfp(run, probe_id, work_dir, settings, args.force)
            print(f"{run.name} {probe_id}: LFP {lfp.get_num_channels()} channels, {lfp.get_total_duration():.0f} s")
            if config is not None and config.line_noise.enabled:
                from ephysalis.processing.linenoise import clean_lfp
                from ephysalis.processing.lfp import lfp_folder
                from ephysalis.timeline.clocks import synchronize_run
                from ephysalis.timeline.steps import aux_channel_map

                sync = synchronize_run(run, aux_channel_map(run, config), cache=SyncEdgeCache(work_dir / "sync"))
                status = clean_lfp(run, probe_id, lfp_folder(work_dir, run.name, probe_id), config, sync)
                print(f"{run.name} {probe_id}: mains removal {'applied' if status['applied'] else 'not applied: ' + status['reason']}")
    _record_step("lfp", args, recording_session, work_dir, config, config_path)
    print(f"Output: {work_dir / 'lfp'}")
    return 0


def _mua(args: argparse.Namespace) -> int:
    from ephysalis.processing.mua import extract_mua
    from ephysalis.timeline.steps import default_work_dir

    recording_session, work_dir, config, config_path = _open_with_config(args)
    _configure(recording_session, config, config_path, work_dir)
    settings = step_settings_from_args("mua", args)
    for run in recording_session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        for probe_id in settings.probes or list(run.probes):
            mua = extract_mua(run, probe_id, work_dir, settings, args.force)
            print(f"{run.name} {probe_id}: MUA {mua.get_num_channels()} channels, {mua.get_total_duration():.0f} s")
    _record_step("mua", args, recording_session, work_dir, config, config_path)
    print(f"Output: {work_dir / 'mua'}")
    return 0


def _export(args: argparse.Namespace) -> int:
    """NWB or DH5 file of a session (or one per run, ``--per-run``), optionally in a BIDS (NWB)
    or BIDS-like (DH5) dataset."""
    from ephysalis.formats.output.nwb import included_runs, single_run_recording_session
    from ephysalis.timeline.steps import default_work_dir

    recording_session, work_dir, config, config_path = _open_with_config(args)
    _configure(recording_session, config, config_path, work_dir)
    print(f"Config: {config_path or 'none'}")
    extension = f".{args.output_format}"
    subject = session_label = ""
    if args.bids:
        from ephysalis.formats.output.bids import bids_labels

        subject, session_label = bids_labels(
            config.subject_id if config else recording_session.name, config.session_id if config else recording_session.name)
        subject, session_label = args.subject or subject, args.session_label or session_label
    if args.per_run and args.existing:
        print("--existing takes one session file, not --per-run", file=sys.stderr)
        return 1
    # (session to write, BIDS run index or None, BIDS task, output path)
    files: list[tuple[RecordingSession, int | None, str | None, Path]] = []
    if args.per_run:
        for number, run in enumerate(included_runs(recording_session, config), start=1):  # BIDS run: position among the included
            task = config.task_of([run.name]) if config else None
            if args.bids:
                from ephysalis.formats.output.bids import ecephys_path

                output = ecephys_path(Path(args.bids), subject, session_label, extension, run=number, task=task)
            else:
                base = Path(args.output) if args.output else work_dir / f"{recording_session.name}{extension}"
                output = base.with_name(f"{base.stem}_{run.name}{extension}")
            files.append((single_run_recording_session(recording_session, run), number, task, output))
    else:
        task = config.task_of([r.name for r in included_runs(recording_session, config)]) if config else None
        if args.bids:
            from ephysalis.formats.output.bids import ecephys_path

            output = ecephys_path(Path(args.bids), subject, session_label, extension, task=task)
        else:
            output = Path(args.output) if args.output else work_dir / f"{recording_session.name}{extension}"
        files.append((recording_session, None, task, output))
    processing = None
    if not args.existing:
        from ephysalis.provenance import processing_record

        processing = processing_record(work_dir, _record_step("export", args, recording_session, work_dir, config, config_path))
    syncs = _included_run_syncs(recording_session, config, work_dir) if not args.existing else {}
    # the runner's local work dir (ephysalis.conversion_batch.source_cache): a file for the dataset is made there and
    # copied in one go (a network file system writes it faster than the many small writes of HDF5)
    from ephysalis.conversion_batch.source_cache import WORK_DIR_ORIGIN_ENV

    build_locally = bool(args.bids and os.environ.get(WORK_DIR_ORIGIN_ENV))
    for file_recording_session, run_index, task, output in files:
        existing = Path(args.existing) if args.existing else None
        built = None
        if existing is None:
            built = work_dir / "export" / output.name if build_locally else output
            if args.output_format == "nwb":
                from ephysalis.formats.output.nwb import write_recording_session_nwb

                write_recording_session_nwb(file_recording_session, work_dir, built, config, syncs, lfp=args.lfp,
                                  compression="gzip" if args.compression else None, processing=processing,
                                  raw=args.raw, run_clock=args.per_run)
            else:
                from ephysalis.formats.output.dh5 import write_recording_session_dh5

                write_recording_session_dh5(file_recording_session, work_dir, built, config, syncs, lfp=args.lfp,
                                  processing=processing, raw=args.raw, run_clock=args.per_run)
            existing = built
        if args.bids:
            from ephysalis.formats.output.bids import add_dh5_to_bids, add_nwb_to_bids

            options = {"power_line_frequency": args.power_line_frequency_hz, "dataset_name": args.dataset_name or None}
            if args.output_format == "nwb":
                add_nwb_to_bids(existing, Path(args.bids), subject, session_label, run=run_index, task=task, **options)
            else:
                add_dh5_to_bids(file_recording_session, work_dir, existing, Path(args.bids), subject, session_label, config,
                                run=run_index, task=task, **options)
        if built is not None and built != output:  # copied into the dataset: not kept in the work dir
            from ephysalis.formats.output.dh5 import sidecar_paths

            for path in [built, *(sidecar_paths(built).values() if args.output_format == "dh5" else [])]:
                path.unlink(missing_ok=True)
        print(f"Output: {output}")
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    """Redo the processing of an NWB/DH5 file from its processing record."""
    import subprocess

    from ephysalis.provenance import (
        compare_environment, compare_inputs, input_fingerprint, read_processing_record, write_replay_configs,
    )

    record = read_processing_record(args.file)
    if record is None:
        print(f"{args.file} has no ephysalis processing record", file=sys.stderr)
        return 1
    export = record["export"]
    recording_session = Path(args.recording_session or export["recording_session"])
    work_dir = Path(args.work_dir)
    if work_dir.resolve() == Path(record["work_dir"]).resolve():
        print("--work-dir must differ from the recorded work dir (its outputs would be reused)", file=sys.stderr)
        return 1
    for line in compare_environment(export.get("environment", {})):
        print(f"WARNING environment: {line}", file=sys.stderr)
    if recording_session.is_dir():
        for line in compare_inputs(export.get("inputs", {}), input_fingerprint(recording_session, [work_dir])):
            print(f"WARNING inputs: {line}", file=sys.stderr)
    else:
        print(f"WARNING: session {recording_session} not found (use --recording-session)", file=sys.stderr)
    commands = write_replay_configs(record, recording_session, work_dir)
    for argv in commands:
        print("ephysalis " + " ".join(f'"{a}"' if " " in a else a for a in argv))
    if not args.run:
        print("(dry run: add --run to run these commands)")
        return 0
    for argv in commands:
        code = subprocess.call([sys.executable, "-u", "-m", "ephysalis.cli.main", *argv])
        if code != 0:
            print(f"failed: ephysalis {' '.join(argv)}", file=sys.stderr)
            return code
    return 0


def add_recording_session_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("inspect", help="Show runs, probes, streams and experimental protocols of a session")
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--conditions", action="store_true", help="Also load the trial conditions")
    p.add_argument(
        "--timing", action="store_true",
        help="Also sync the streams and detect trials and running speed (uses the sync edge cache)",
    )
    p.add_argument("--cache-dir", help="Sync edge cache (default: <session>/ephysalis/sync)")
    p.set_defaults(func=per_recording_session(_inspect))

    p = sub.add_parser(
        "align-clocks",
        help="Put every stream of a run on the reference clock: extract the sync-pulse edges of all streams "
        "(cached; reads the full AP files) and match them; writes <work-dir>/clocks/<run>.json",
    )
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {RECORDING_SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder; the cache goes to <work-dir>/sync (default: <session>/ephysalis)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    add_step_settings_arguments(p, "align-clocks")
    p.add_argument("--force", action="store_true", help="Re-extract even if cached")
    p.set_defaults(func=per_recording_session(_align_clocks))

    p = sub.add_parser(
        "spike-sort",
        help="Spike sort each probe over all included runs ([spike_sorting] in session.toml, default Kilosort4) "
        "and cut the spikes into runs",
    )
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {RECORDING_SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder with sync/, sorting/, spikes/ (default: <session>/ephysalis)")
    add_step_settings_arguments(p, "spike-sort")
    p.set_defaults(func=per_recording_session(_spike_sort))

    for name, func, help_text, outputs in (
        ("extract-trials", _extract_trials, "Write the trial tables of all experimental protocols of the metadata plugin and the "
         "run start times from its clock (on the reference clock)", "trials/, run_start/"),
        ("extract-events", _extract_events, "Write the edges of the digital lines and the event words of every run "
         "(on the reference clock)", "events/"),
        ("extract-behavior", _extract_behavior, "Write the eye position (1 kHz, V) and running speed (1 kHz, cm/s) of "
         "every run (on the reference clock)", "eye/, running_speed/"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
        p.add_argument("--format", help="Input format (default: auto-detect)")
        p.add_argument("--config", help=f"Session config (default: {RECORDING_SESSION_CONFIG_FILENAME} in the session or work dir)")
        p.add_argument("--work-dir", help=f"Output folder with {outputs} (default: <session>/ephysalis)")
        p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
        add_step_settings_arguments(p, name)
        p.set_defaults(func=per_recording_session(func))

    p = sub.add_parser("lfp", help="Write the LFP of every run and probe, downsampled with an anti-alias filter")
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {RECORDING_SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder, LFP goes to <work-dir>/lfp (default: <session>/ephysalis)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    add_step_settings_arguments(p, "lfp")
    p.add_argument("--force", action="store_true", help="Overwrite existing LFP")
    p.set_defaults(func=per_recording_session(_lfp))

    p = sub.add_parser("calibrate", help="Channel gains from the calibration-sine run ([calibration] in session.toml)")
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {RECORDING_SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder, calibration.json goes there (default: <session>/ephysalis)")
    add_step_settings_arguments(p, "calibrate")
    p.set_defaults(func=per_recording_session(_calibrate))

    p = sub.add_parser("quality", help="Select the good channels of every run and probe with [probes.<id>.quality] "
                       "(from the LFP; the CSD then uses them)")
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {RECORDING_SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder with lfp/, results go to <work-dir>/quality (default: <session>/ephysalis)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    add_step_settings_arguments(p, "quality")
    p.set_defaults(func=per_recording_session(_quality))

    p = sub.add_parser("csd", help="Write the CSD of every run and probe with [probes.<id>.csd] (from the LFP)")
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {RECORDING_SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder with lfp/, CSD goes to <work-dir>/csd (default: <session>/ephysalis)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    add_step_settings_arguments(p, "csd")
    p.add_argument("--force", action="store_true", help="Overwrite existing CSD")
    p.set_defaults(func=per_recording_session(_csd))

    p = sub.add_parser("mua", help="Write the MUA envelope (high-pass, rectified, resampled) of every run and probe")
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {RECORDING_SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder, MUA goes to <work-dir>/mua (default: <session>/ephysalis)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    add_step_settings_arguments(p, "mua")
    p.add_argument("--force", action="store_true", help="Overwrite existing MUA")
    p.set_defaults(func=per_recording_session(_mua))

    p = sub.add_parser(
        "export",
        help="Write one NWB or DH5 file with all runs and probes from the outputs in the work dir, "
        "optionally into a BIDS dataset",
    )
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {RECORDING_SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Folder with the outputs of the other steps (default: <session>/ephysalis)")
    add_step_settings_arguments(p, "export")
    p.set_defaults(func=per_recording_session(_export))

    p = sub.add_parser(
        "reproduce",
        help="Redo the processing of an NWB/DH5 file from its processing record (configs, options, versions)",
    )
    p.add_argument("file", type=Path, help="NWB or DH5 file written by `ephysalis export`")
    p.add_argument("--work-dir", required=True, help="New work dir (gets the recorded session.toml)")
    p.add_argument("--recording-session", help="Recording session folder (default: the recorded path)")
    p.add_argument("--run", action="store_true", help="Run the commands (default: only print them)")
    p.set_defaults(func=_reproduce)

    p = sub.add_parser("init", help=f"Create {RECORDING_SESSION_CONFIG_FILENAME} for a session")
    p.add_argument("recording_sessions", nargs="+", type=Path, metavar="recording_session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--metadata", help="Metadata file outside the session folder for the metadata plugins, e.g. the lab "
                   "workbook of a lab setup")
    p.add_argument("-o", "--output", help=f"Output path (default: <session>/{RECORDING_SESSION_CONFIG_FILENAME})")
    p.add_argument("--force", action="store_true", help="Overwrite an existing config")
    p.set_defaults(func=per_recording_session(_init))
