"""Pipeline commands: ``ephyscon inspect``, ``init``, ``sync``, ``timing``, ``lfp``, ``mua``, ``sort``,
``export`` and ``reproduce``.

Every step takes one or more sessions. With several sessions, per-session paths contain the
placeholder ``{session}`` (the session folder name), e.g. ``--work-dir D:/work/{session}``;
without ``--work-dir`` each session uses ``<session>/ephyscon``. A failing session does not
stop the others; the exit code is 1 if any session failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from ephyscon.session import Run, Session
from ephyscon.session_config import (
    SESSION_CONFIG_FILENAME,
    SessionConfig,
    default_session_config,
    load_session_config,
    save_session_config,
)
from ephyscon.io.base import open_session
from ephyscon.io.probes import configure_probes
from ephyscon.sync import RunSync, SyncEdgeCache
from ephyscon.stimulus.base import detect_stimulus_system, get_stimulus_system

logger = logging.getLogger(__name__)
SESSION_PLACEHOLDER = "{session}"
# options that hold a path of one session: need SESSION_PLACEHOLDER with several sessions
_PER_SESSION_OPTIONS = ("work_dir", "config", "cache_dir", "output", "existing")


def per_session(func: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
    """Run a one-session command for every session in ``args.sessions``."""

    def run(args: argparse.Namespace) -> int:
        sessions = [Path(s) for s in args.sessions]
        if len(sessions) > 1:
            fixed = [f"--{o.replace('_', '-')}" for o in _PER_SESSION_OPTIONS
                     if getattr(args, o, None) and SESSION_PLACEHOLDER not in str(getattr(args, o))]
            if fixed:
                print(f"{', '.join(fixed)} with several sessions needs the placeholder {SESSION_PLACEHOLDER}, "
                      f"e.g. --work-dir D:/work/{SESSION_PLACEHOLDER}", file=sys.stderr)
                return 2
        failed = []
        for session in sessions:
            one = argparse.Namespace(**vars(args))
            one.session = session
            for option in _PER_SESSION_OPTIONS:
                value = getattr(one, option, None)
                if value is not None:
                    setattr(one, option, str(value).replace(SESSION_PLACEHOLDER, session.name))
            if len(sessions) > 1:
                print(f"=== {session.name} ===", flush=True)
            try:
                code = func(one)
            except KeyboardInterrupt:
                raise
            except Exception:
                if len(sessions) == 1:
                    raise
                logger.exception("%s failed", session.name)
                code = 1
            if code:
                failed.append(session.name)
        if len(sessions) > 1:
            print(f"{len(sessions) - len(failed)} of {len(sessions)} sessions done"
                  + (f"; failed: {', '.join(failed)}" if failed else ""))
        return 1 if failed else 0

    return run


def _inspect(args: argparse.Namespace) -> int:
    session = open_session(args.session, args.format)
    config_path = session.path / SESSION_CONFIG_FILENAME
    config = load_session_config(config_path) if config_path.exists() else None
    configure_probes(session, config, config_path.parent)

    print(f"Session {session.name} ({session.format.name}), {len(session.runs)} runs")
    print(f"Config: {config_path if config else 'none (run `ephyscon init`)'}")
    for run in session.runs:
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
        others = [s for s in run.extra.get("streams", []) if s != run.aux_stream
                  and all(s not in info.streams.values() for info in run.probes.values())]
        if others:  # e.g. for probes defined in session.toml
            print(f"      other streams: {', '.join(others)}")

        system = (
            get_stimulus_system(run_config.stimulus_system)
            if run_config and run_config.stimulus_system
            else detect_stimulus_system(run)
        )
        if system is None:
            print("      stimulus: none found")
        else:
            for paradigm in system.discover(run):
                count = paradigm.metadata.get("stimulusCount", "?")
                line = f"      stimulus [{system.name}] {paradigm.name}: {count} trials"
                if args.conditions:
                    conditions = system.load_conditions(paradigm)
                    line += f", {conditions['condition'].nunique()} conditions"
                print(line)
        if args.timing and (run_config is None or run_config.include):
            _print_timing(run, config, _sync_cache(session.path, args.cache_dir))
    return 0


def _sync_cache(session_path: Path, cache_dir: str | None) -> SyncEdgeCache:
    from ephyscon.timing import default_sync_cache

    return SyncEdgeCache(cache_dir) if cache_dir else default_sync_cache(session_path)


def _print_timing(run: Run, config: SessionConfig | None, cache: SyncEdgeCache) -> None:
    from ephyscon.timing import extract_run_timing

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
    session = open_session(args.session, args.format)
    output = Path(args.output) if args.output else session.path / SESSION_CONFIG_FILENAME
    if output.exists() and not args.force:
        print(f"{output} exists, use --force to overwrite", file=sys.stderr)
        return 1

    if args.excel or (session.path / "Depths.txt").exists():
        from ephyscon.migrate import migrate_session

        result = migrate_session(session, Path(args.excel) if args.excel else None, args.sheet)
        config, warnings = result.config, result.warnings
    else:
        config, warnings = default_session_config(session), []

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    save_session_config(config, output)
    print(f"Wrote {output}")
    return 0


def _sync(args: argparse.Namespace) -> int:
    from ephyscon.sync import load_or_extract_edges
    from ephyscon.timing import WORK_DIR_NAME, aux_channel_map

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    _configure(session, config, config_path, work_dir)
    cache = SyncEdgeCache(args.cache_dir) if args.cache_dir else SyncEdgeCache(work_dir / "sync")
    for run in session.runs:
        run_config = config.run_config(run.name) if config else None
        if args.runs and run.name not in args.runs:
            continue
        if not args.runs and run_config is not None and not run_config.include:
            continue
        if args.force and cache.path(run.name).exists():
            cache.path(run.name).unlink()
        edges = load_or_extract_edges(run, aux_channel_map(run, config), cache)
        counts = ", ".join(f"{stream} {e.samples.size}" for stream, e in edges.items())
        print(f"{run.name}: sync edges {counts}")
    _record_step("sync", args, session, work_dir, config, config_path)
    print(f"Cache: {cache.directory}")
    return 0


def _record_step(step: str, args: argparse.Namespace, session: Session, work_dir: Path,
                 config: SessionConfig | None, config_path: Path | None) -> dict:
    """Append this invocation to ``<work dir>/provenance/<step>.json``; returns its record."""
    from ephyscon.provenance import step_record, write_step_record

    options = {k: v for k, v in vars(args).items()
               if k not in ("func", "command", "session", "sessions", "argv", "verbose") and not callable(v)}
    record = step_record(step, session, config, config_path, options, getattr(args, "argv", None), work_dir)
    write_step_record(work_dir, record)
    return record


def _parse_value(text: str) -> object:
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _included_run_syncs(session: Session, config: SessionConfig | None, work_dir: Path) -> dict[str, RunSync]:
    """RunSync of every included run, from the sync edge cache in the work dir."""
    from ephyscon.sync import synchronize_run
    from ephyscon.timing import aux_channel_map

    cache = SyncEdgeCache(work_dir / "sync")
    syncs = {}
    for run in session.runs:
        run_config = config.run_config(run.name) if config else None
        if run_config is None or run_config.include:
            syncs[run.name] = synchronize_run(run, aux_channel_map(run, config), cache=cache)
    return syncs


def _sort(args: argparse.Namespace) -> int:
    import tempfile

    from ephyscon.sorting import sort_probe
    from ephyscon.timing import WORK_DIR_NAME

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    _configure(session, config, config_path, work_dir)
    print(f"Config: {config_path or 'none (sorting all channels)'}")
    syncs = _included_run_syncs(session, config, work_dir)
    params = dict(kv.split("=", 1) for kv in args.param or [])
    scratch = Path(args.scratch_dir) if args.scratch_dir else Path(tempfile.gettempdir()) / "ephyscon"
    for probe_id in args.probes or session.probe_ids:
        runs = sort_probe(
            session, probe_id, work_dir, scratch, config, syncs,
            sorter_params={k: _parse_value(v) for k, v in params.items()},
            n_jobs=args.n_jobs, keep_binary=args.keep_binary, sorter=args.sorter,
        )
        for run_name, spikes in runs.items():
            print(f"{probe_id} {run_name}: {len(spikes)} spikes, {spikes.unit_id.nunique()} units")
    _record_step("sort", args, session, work_dir, config, config_path)
    print(f"Output: {work_dir}")
    return 0


def _configure(session: Session, config: SessionConfig | None, config_path: Path | None, work_dir: Path) -> None:
    """Probes from session.toml and the calibration (``[calibration]``, ``ephyscon calibrate``)."""
    from ephyscon.calibration import apply_calibration, load_calibration

    config_dir = config_path.parent if config_path else None
    configure_probes(session, config, config_dir)
    apply_calibration(session, load_calibration(config, config_dir, work_dir))


def _calibrate(args: argparse.Namespace) -> int:
    from ephyscon.calibration import compute_calibration, write_calibration
    from ephyscon.timing import WORK_DIR_NAME

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    if config is None or config.calibration.run is None:
        print("Nothing to do: [calibration] run is not set in session.toml")
        return 0
    configure_probes(session, config, config_path.parent if config_path else None)
    calibration = compute_calibration(session, config)
    path = write_calibration(calibration, work_dir)
    for probe_id, entry in calibration["probes"].items():
        gains = [g for g in entry["gains_uv"] if g is not None]
        print(f"{probe_id}: sine {entry['frequency_hz']:.3f} Hz in {entry['run']}, {len(gains)} channels calibrated "
              f"(median {sorted(gains)[len(gains) // 2] if gains else float('nan'):.4f} uV per count), "
              f"without signal: {', '.join(entry['without_signal']) or 'none'}")
    _record_step("calibrate", args, session, work_dir, config, config_path)
    print(f"Output: {path}")
    return 0


def _csd(args: argparse.Namespace) -> int:
    from ephyscon.csd import extract_csd
    from ephyscon.timing import WORK_DIR_NAME

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    _configure(session, config, config_path, work_dir)
    probes = {pid: p.csd for pid, p in (config.probes.items() if config else []) if p.csd is not None}
    if not probes:
        print("Nothing to do: no probe has [probes.<id>.csd] in session.toml")
        return 0
    for run in session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        for probe_id, csd_config in probes.items():
            if probe_id in run.probes and (not args.probes or probe_id in args.probes):
                csd = extract_csd(run, probe_id, work_dir, csd_config, args.n_jobs, args.force)
                print(f"{run.name} {probe_id}: CSD {csd.get_num_channels()} channels, {csd.get_total_duration():.0f} s")
    _record_step("csd", args, session, work_dir, config, config_path)
    print(f"Output: {work_dir / 'csd'}")
    return 0


def _load_config_for(session_path: Path, work_dir: Path, config: str | None) -> tuple[SessionConfig | None, Path | None]:
    candidates = [Path(config)] if config else [session_path / SESSION_CONFIG_FILENAME, work_dir / SESSION_CONFIG_FILENAME]
    path = next((p for p in candidates if p.exists() or config), None)
    return (load_session_config(path) if path else None), path


def _timing(args: argparse.Namespace) -> int:
    import numpy as np

    from ephyscon.timing import WORK_DIR_NAME, extract_run_timing

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    _configure(session, config, config_path, work_dir)
    cache = SyncEdgeCache(work_dir / "sync")
    (work_dir / "trials").mkdir(parents=True, exist_ok=True)
    (work_dir / "running_speed").mkdir(exist_ok=True)
    (work_dir / "eye").mkdir(exist_ok=True)
    (work_dir / "events").mkdir(exist_ok=True)
    for run in session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        timing = extract_run_timing(run, config, cache)
        for paradigm, trials in timing.trials.items():
            trials.to_csv(work_dir / "trials" / f"{run.name}_{paradigm}.csv", index=False)
            print(f"{run.name} {paradigm}: {len(trials)} trials")
        if timing.running_speed is not None:
            np.save(work_dir / "running_speed" / f"{run.name}.npy", timing.running_speed.astype(np.float32))
        if timing.eye_position is not None:
            np.save(work_dir / "eye" / f"{run.name}.npy", timing.eye_position)
        if timing.events is not None:
            timing.events.to_csv(work_dir / "events" / f"{run.name}.csv", index=False)
            print(f"{run.name}: {len(timing.events)} events")
    _record_step("timing", args, session, work_dir, config, config_path)
    print(f"Output: {work_dir}")
    return 0


def _lfp(args: argparse.Namespace) -> int:
    from ephyscon.lfp import extract_lfp
    from ephyscon.timing import WORK_DIR_NAME

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    _configure(session, config, config_path, work_dir)
    for run in session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        for probe_id in args.probes or list(run.probes):
            lfp = extract_lfp(run, probe_id, work_dir, args.rate, args.n_jobs, args.force)
            print(f"{run.name} {probe_id}: LFP {lfp.get_num_channels()} channels, {lfp.get_total_duration():.0f} s")
            if config is not None and config.line_noise.enabled:
                from ephyscon.linenoise import clean_lfp
                from ephyscon.lfp import lfp_folder
                from ephyscon.sync import synchronize_run
                from ephyscon.timing import aux_channel_map

                sync = synchronize_run(run, aux_channel_map(run, config), cache=SyncEdgeCache(work_dir / "sync"))
                status = clean_lfp(run, probe_id, lfp_folder(work_dir, run.name, probe_id), config, sync)
                print(f"{run.name} {probe_id}: mains removal {'applied' if status['applied'] else 'not applied: ' + status['reason']}")
    _record_step("lfp", args, session, work_dir, config, config_path)
    print(f"Output: {work_dir / 'lfp'}")
    return 0


def _mua(args: argparse.Namespace) -> int:
    from ephyscon.mua import extract_mua
    from ephyscon.timing import WORK_DIR_NAME

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    _configure(session, config, config_path, work_dir)
    for run in session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        for probe_id in args.probes or list(run.probes):
            mua = extract_mua(run, probe_id, work_dir, args.rate, args.n_jobs, args.force)
            print(f"{run.name} {probe_id}: MUA {mua.get_num_channels()} channels, {mua.get_total_duration():.0f} s")
    _record_step("mua", args, session, work_dir, config, config_path)
    print(f"Output: {work_dir / 'mua'}")
    return 0


def _export(args: argparse.Namespace) -> int:
    """NWB or DH5 file of a session, optionally in a BIDS (NWB) or BIDS-like (DH5) dataset."""
    from ephyscon.timing import WORK_DIR_NAME

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    _configure(session, config, config_path, work_dir)
    print(f"Config: {config_path or 'none'}")
    extension = f".{args.output_format}"
    if args.bids:
        from ephyscon.bids import bids_labels, ecephys_path

        subject, session_label = bids_labels(
            config.subject_id if config else session.name, config.session_id if config else session.name)
        subject, session_label = args.subject or subject, args.session_label or session_label
        output = ecephys_path(Path(args.bids), subject, session_label, extension)
    else:
        output = Path(args.output) if args.output else work_dir / f"{session.name}{extension}"
    existing = Path(args.existing) if args.existing else None
    if existing is None:
        from ephyscon.provenance import processing_record

        processing = processing_record(work_dir, _record_step("export", args, session, work_dir, config, config_path))
        syncs = _included_run_syncs(session, config, work_dir)
        if args.output_format == "nwb":
            from ephyscon.nwb import write_session_nwb

            write_session_nwb(session, work_dir, output, config, syncs, lfp=not args.no_lfp,
                              compression=None if args.no_compression else "gzip", processing=processing,
                              raw=args.raw)
        else:
            from ephyscon.dh5 import write_session_dh5

            write_session_dh5(session, work_dir, output, config, syncs, lfp=not args.no_lfp,
                              processing=processing, raw=args.raw)
        existing = output
    if args.bids:
        from ephyscon.bids import add_dh5_to_bids, add_nwb_to_bids

        options = {"power_line_frequency": args.power_line_frequency, "dataset_name": args.dataset_name}
        if args.output_format == "nwb":
            add_nwb_to_bids(existing, Path(args.bids), subject, session_label, **options)
        else:
            add_dh5_to_bids(session, work_dir, existing, Path(args.bids), subject, session_label, config, **options)
    print(f"Output: {output}")
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    """Redo the processing of an NWB/DH5 file from its processing record."""
    import subprocess

    from ephyscon.provenance import (
        compare_environment, compare_inputs, input_fingerprint, read_processing_record, write_replay_configs,
    )

    record = read_processing_record(args.file)
    if record is None:
        print(f"{args.file} has no ephyscon processing record", file=sys.stderr)
        return 1
    export = record["export"]
    session = Path(args.session or export["session"])
    work_dir = Path(args.work_dir)
    if work_dir.resolve() == Path(record["work_dir"]).resolve():
        print("--work-dir must differ from the recorded work dir (its outputs would be reused)", file=sys.stderr)
        return 1
    for line in compare_environment(export.get("environment", {})):
        print(f"WARNING environment: {line}", file=sys.stderr)
    if session.is_dir():
        for line in compare_inputs(export.get("inputs", {}), input_fingerprint(session, [work_dir])):
            print(f"WARNING inputs: {line}", file=sys.stderr)
    else:
        print(f"WARNING: session {session} not found (use --session)", file=sys.stderr)
    commands = write_replay_configs(record, session, work_dir)
    for argv in commands:
        print("ephyscon " + " ".join(f'"{a}"' if " " in a else a for a in argv))
    if not args.run:
        print("(dry run: add --run to run these commands)")
        return 0
    for argv in commands:
        code = subprocess.call([sys.executable, "-u", "-m", "ephyscon.cli.main", *argv])
        if code != 0:
            print(f"failed: ephyscon {' '.join(argv)}", file=sys.stderr)
            return code
    return 0


def add_session_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("inspect", help="Show runs, probes, streams and paradigms of a session")
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--conditions", action="store_true", help="Also load the trial conditions")
    p.add_argument(
        "--timing", action="store_true",
        help="Also sync the streams and detect trials and running speed (uses the sync edge cache)",
    )
    p.add_argument("--cache-dir", help="Sync edge cache (default: <session>/ephyscon/sync)")
    p.set_defaults(func=per_session(_inspect))

    p = sub.add_parser(
        "sync",
        help="Extract the sync edges of all streams once and cache them (reads the full AP files)",
    )
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder; the cache goes to <work-dir>/sync (default: <session>/ephyscon)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    p.add_argument("--cache-dir", help="Sync edge cache (default: <work-dir>/sync)")
    p.add_argument("--force", action="store_true", help="Re-extract even if cached")
    p.set_defaults(func=per_session(_sync))

    p = sub.add_parser(
        "sort",
        help="Spike sort each probe over all included runs ([sorting] in session.toml, default Kilosort4) "
        "and cut the spikes into runs",
    )
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder with sync/, sorting/, spikes/ (default: <session>/ephyscon)")
    p.add_argument("--scratch-dir", help="Fast local folder for the concatenated binary (default: system temp)")
    p.add_argument("--probes", nargs="+", help="Only these probes (default: all)")
    p.add_argument("--sorter", help="spikeinterface sorter, e.g. kilosort4, mountainsort5 (default: session.toml)")
    p.add_argument("--param", nargs="+", metavar="KEY=VALUE", help="Sorter parameters, e.g. batch_size=30000")
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs for copying and waveforms")
    p.add_argument("--keep-binary", action="store_true", help="Keep the local binary copy (e.g. for Phy)")
    p.set_defaults(func=per_session(_sort))

    p = sub.add_parser(
        "timing",
        help="Write trial tables (reference clock), running speed (1 kHz, cm/s) and eye position (1 kHz, V) of every run",
    )
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder with sync/, trials/, running_speed/ (default: <session>/ephyscon)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    p.set_defaults(func=per_session(_timing))

    p = sub.add_parser("lfp", help="Write the LFP of every run and probe, downsampled with an anti-alias filter")
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder, LFP goes to <work-dir>/lfp (default: <session>/ephyscon)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    p.add_argument("--probes", nargs="+", help="Only these probes (default: all)")
    p.add_argument("--rate", type=float, default=1000.0, help="LFP sampling rate in Hz (default 1000)")
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    p.add_argument("--force", action="store_true", help="Overwrite existing LFP")
    p.set_defaults(func=per_session(_lfp))

    p = sub.add_parser("calibrate", help="Channel gains from the calibration-sine run ([calibration] in session.toml)")
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder, calibration.json goes there (default: <session>/ephyscon)")
    p.set_defaults(func=per_session(_calibrate))

    p = sub.add_parser("csd", help="Write the CSD of every run and probe with [probes.<id>.csd] (from the LFP)")
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder with lfp/, CSD goes to <work-dir>/csd (default: <session>/ephyscon)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    p.add_argument("--probes", nargs="+", help="Only these probes (default: all with [probes.<id>.csd])")
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    p.add_argument("--force", action="store_true", help="Overwrite existing CSD")
    p.set_defaults(func=per_session(_csd))

    p = sub.add_parser("mua", help="Write the MUA envelope (300 Hz high-pass, rectified, 1 kHz) of every run and probe")
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder, MUA goes to <work-dir>/mua (default: <session>/ephyscon)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    p.add_argument("--probes", nargs="+", help="Only these probes (default: all)")
    p.add_argument("--rate", type=float, default=1000.0, help="MUA sampling rate in Hz (default 1000)")
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    p.add_argument("--force", action="store_true", help="Overwrite existing MUA")
    p.set_defaults(func=per_session(_mua))

    p = sub.add_parser(
        "export",
        help="Write one NWB or DH5 file with all runs and probes from the outputs in the work dir, "
        "optionally into a BIDS dataset",
    )
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--output-format", choices=["nwb", "dh5"], default="nwb", help="File format (default nwb)")
    p.add_argument("--bids", metavar="ROOT", help="Put the file into this BIDS dataset (BIDS-like for DH5: "
                   "BIDS only allows NWB and NIX data files)")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Folder with the outputs of timing, lfp and sort (default: <session>/ephyscon)")
    p.add_argument("-o", "--output", help="Output file without --bids (default: <work-dir>/<session>.<format>)")
    p.add_argument("--existing", help="With --bids: copy this existing session file instead of writing a new one")
    p.add_argument("--subject", help="BIDS subject label (default: subject id without non-alphanumerics)")
    p.add_argument("--session-label", help="BIDS session label (default: session id without the subject prefix)")
    p.add_argument("--power-line-frequency", type=float, default=50.0, help="Mains frequency in Hz (default 50)")
    p.add_argument("--dataset-name", help="Name in dataset_description.json (default: folder name)")
    p.add_argument("--no-lfp", action="store_true", help="Leave out the LFP")
    p.add_argument("--raw", action="store_true", help="Include the raw data (AP/wideband streams and analog aux "
                   "inputs; about the size of the recording)")
    p.add_argument("--no-compression", action="store_true", help="NWB: write the LFP without gzip compression")
    p.set_defaults(func=per_session(_export))

    p = sub.add_parser(
        "reproduce",
        help="Redo the processing of an NWB/DH5 file from its processing record (configs, options, versions)",
    )
    p.add_argument("file", type=Path, help="NWB or DH5 file written by `ephyscon export`")
    p.add_argument("--work-dir", required=True, help="New work dir (gets the recorded session.toml)")
    p.add_argument("--session", help="Session folder (default: the recorded path)")
    p.add_argument("--run", action="store_true", help="Run the commands (default: only print them)")
    p.set_defaults(func=_reproduce)

    p = sub.add_parser("init", help=f"Create {SESSION_CONFIG_FILENAME} for a session")
    p.add_argument("sessions", nargs="+", type=Path, metavar="session")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--excel", help="Legacy lab Excel workbook to migrate run/probe info from")
    p.add_argument("--sheet", help="Sheet name in the workbook (default: session folder name)")
    p.add_argument("-o", "--output", help=f"Output path (default: <session>/{SESSION_CONFIG_FILENAME})")
    p.add_argument("--force", action="store_true", help="Overwrite an existing config")
    p.set_defaults(func=per_session(_init))
