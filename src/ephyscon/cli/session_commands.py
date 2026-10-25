"""Session-level commands: ``ephyscon inspect``, ``init``, ``sync``, ``timing``, ``sort``, ``lfp``, ``nwb`` and ``bids``."""

from __future__ import annotations

import argparse
import sys
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
from ephyscon.sync import RunSync, SyncEdgeCache
from ephyscon.stimulus.base import detect_stimulus_system, get_stimulus_system


def _inspect(args: argparse.Namespace) -> int:
    session = open_session(args.session, args.format)
    config_path = session.path / SESSION_CONFIG_FILENAME
    config = load_session_config(config_path) if config_path.exists() else None

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
    from ephyscon.timing import aux_channel_map

    session = open_session(args.session, args.format)
    config_path = session.path / SESSION_CONFIG_FILENAME
    config = load_session_config(config_path) if config_path.exists() else None
    cache = _sync_cache(session.path, args.cache_dir)
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
    print(f"Cache: {cache.directory}")
    return 0


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
    print(f"Config: {config_path or 'none (sorting all channels)'}")
    syncs = _included_run_syncs(session, config, work_dir)
    params = dict(kv.split("=", 1) for kv in args.param or [])
    scratch = Path(args.scratch_dir) if args.scratch_dir else Path(tempfile.gettempdir()) / "ephyscon"
    for probe_id in args.probes or session.probe_ids:
        runs = sort_probe(
            session, probe_id, work_dir, scratch, config, syncs,
            sorter_params={k: _parse_value(v) for k, v in params.items()},
            n_jobs=args.n_jobs, keep_binary=args.keep_binary,
        )
        for run_name, spikes in runs.items():
            print(f"{probe_id} {run_name}: {len(spikes)} spikes, {spikes.unit_id.nunique()} units")
    print(f"Output: {work_dir}")
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
    config, _ = _load_config_for(session.path, work_dir, args.config)
    cache = SyncEdgeCache(work_dir / "sync")
    (work_dir / "trials").mkdir(parents=True, exist_ok=True)
    (work_dir / "running_speed").mkdir(exist_ok=True)
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
    print(f"Output: {work_dir}")
    return 0


def _lfp(args: argparse.Namespace) -> int:
    from ephyscon.lfp import extract_lfp
    from ephyscon.timing import WORK_DIR_NAME

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, _ = _load_config_for(session.path, work_dir, args.config)
    for run in session.runs:
        run_config = config.run_config(run.name) if config else None
        if (args.runs and run.name not in args.runs) or (not args.runs and run_config is not None and not run_config.include):
            continue
        for probe_id in args.probes or list(run.probes):
            lfp = extract_lfp(run, probe_id, work_dir, args.rate, args.n_jobs, args.force)
            print(f"{run.name} {probe_id}: LFP {lfp.get_num_channels()} channels, {lfp.get_total_duration():.0f} s")
    print(f"Output: {work_dir / 'lfp'}")
    return 0


def _nwb(args: argparse.Namespace) -> int:
    from ephyscon.nwb import write_session_nwb
    from ephyscon.timing import WORK_DIR_NAME

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    print(f"Config: {config_path or 'none'}")
    output = Path(args.output) if args.output else work_dir / f"{session.name}.nwb"
    syncs = _included_run_syncs(session, config, work_dir) if not args.no_lfp else {}
    write_session_nwb(session, work_dir, output, config, syncs, lfp=not args.no_lfp,
                      compression=None if args.no_compression else "gzip")
    print(f"Output: {output}")
    return 0


def _bids(args: argparse.Namespace) -> int:
    from ephyscon.bids import add_nwb_to_bids, bids_labels, ecephys_path, write_bids_sidecars
    from ephyscon.nwb import write_session_nwb
    from ephyscon.timing import WORK_DIR_NAME

    session = open_session(args.session, args.format)
    work_dir = Path(args.work_dir) if args.work_dir else session.path / WORK_DIR_NAME
    config, config_path = _load_config_for(session.path, work_dir, args.config)
    print(f"Config: {config_path or 'none'}")
    subject, session_label = bids_labels(
        config.subject_id if config else session.name, config.session_id if config else session.name)
    subject, session_label = args.subject or subject, args.session_label or session_label
    root = Path(args.bids_root)
    options = {"power_line_frequency": args.power_line_frequency, "dataset_name": args.dataset_name}
    if args.nwb:
        output = add_nwb_to_bids(Path(args.nwb), root, subject, session_label, **options)
    else:
        output = ecephys_path(root, subject, session_label)
        syncs = _included_run_syncs(session, config, work_dir) if not args.no_lfp else {}
        write_session_nwb(session, work_dir, output, config, syncs, lfp=not args.no_lfp,
                          compression=None if args.no_compression else "gzip")
        write_bids_sidecars(output, root, subject, session_label, **options)
    print(f"Output: {output}")
    return 0


def add_session_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("inspect", help="Show runs, probes, streams and paradigms of a session")
    p.add_argument("session", type=Path)
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--conditions", action="store_true", help="Also load the trial conditions")
    p.add_argument(
        "--timing", action="store_true",
        help="Also sync the streams and detect trials and running speed (uses the sync edge cache)",
    )
    p.add_argument("--cache-dir", help="Sync edge cache (default: <session>/ephyscon/sync)")
    p.set_defaults(func=_inspect)

    p = sub.add_parser(
        "sync",
        help="Extract the sync edges of all streams once and cache them (reads the full AP files)",
    )
    p.add_argument("session", type=Path)
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    p.add_argument("--cache-dir", help="Sync edge cache (default: <session>/ephyscon/sync)")
    p.add_argument("--force", action="store_true", help="Re-extract even if cached")
    p.set_defaults(func=_sync)

    p = sub.add_parser(
        "sort",
        help="Spike sort each probe over all included runs (Kilosort4) and cut the spikes into runs",
    )
    p.add_argument("session", type=Path)
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder with sync/, sorting/, spikes/ (default: <session>/ephyscon)")
    p.add_argument("--scratch-dir", help="Fast local folder for the concatenated binary (default: system temp)")
    p.add_argument("--probes", nargs="+", help="Only these probes (default: all)")
    p.add_argument("--param", nargs="+", metavar="KEY=VALUE", help="Kilosort4 parameters, e.g. batch_size=30000")
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs for copying and waveforms")
    p.add_argument("--keep-binary", action="store_true", help="Keep the local binary copy (e.g. for Phy)")
    p.set_defaults(func=_sort)

    p = sub.add_parser(
        "timing",
        help="Write trial tables (reference clock) and running speed (1 kHz, cm/s) of every run",
    )
    p.add_argument("session", type=Path)
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder with sync/, trials/, running_speed/ (default: <session>/ephyscon)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    p.set_defaults(func=_timing)

    p = sub.add_parser("lfp", help="Write the LFP of every run and probe, downsampled with an anti-alias filter")
    p.add_argument("session", type=Path)
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Output folder, LFP goes to <work-dir>/lfp (default: <session>/ephyscon)")
    p.add_argument("--runs", nargs="+", help="Only these runs (default: all included runs)")
    p.add_argument("--probes", nargs="+", help="Only these probes (default: all)")
    p.add_argument("--rate", type=float, default=1000.0, help="LFP sampling rate in Hz (default 1000)")
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    p.add_argument("--force", action="store_true", help="Overwrite existing LFP")
    p.set_defaults(func=_lfp)

    p = sub.add_parser("nwb", help="Write one NWB file with all runs and probes from the outputs in the work dir")
    p.add_argument("session", type=Path)
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Folder with the outputs of timing, lfp and sort (default: <session>/ephyscon)")
    p.add_argument("-o", "--output", help="NWB file (default: <work-dir>/<session>.nwb)")
    p.add_argument("--no-lfp", action="store_true", help="Leave out the LFP")
    p.add_argument("--no-compression", action="store_true", help="Write the LFP without gzip compression")
    p.set_defaults(func=_nwb)

    p = sub.add_parser("bids", help="Write the session NWB file and its sidecars into a BIDS (ecephys) dataset")
    p.add_argument("session", type=Path)
    p.add_argument("bids_root", help="BIDS dataset folder (created if needed)")
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--config", help=f"Session config (default: {SESSION_CONFIG_FILENAME} in the session or work dir)")
    p.add_argument("--work-dir", help="Folder with the outputs of timing, lfp and sort (default: <session>/ephyscon)")
    p.add_argument("--nwb", help="Copy this existing session NWB file instead of writing a new one")
    p.add_argument("--subject", help="BIDS subject label (default: subject id without non-alphanumerics)")
    p.add_argument("--session-label", help="BIDS session label (default: session id without the subject prefix)")
    p.add_argument("--power-line-frequency", type=float, default=50.0, help="Mains frequency in Hz (default 50)")
    p.add_argument("--dataset-name", help="Name in dataset_description.json (default: folder name)")
    p.add_argument("--no-lfp", action="store_true", help="Leave out the LFP")
    p.add_argument("--no-compression", action="store_true", help="Write the LFP without gzip compression")
    p.set_defaults(func=_bids)

    p = sub.add_parser("init", help=f"Create {SESSION_CONFIG_FILENAME} for a session")
    p.add_argument("session", type=Path)
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--excel", help="Legacy lab Excel workbook to migrate run/probe info from")
    p.add_argument("--sheet", help="Sheet name in the workbook (default: session folder name)")
    p.add_argument("-o", "--output", help=f"Output path (default: <session>/{SESSION_CONFIG_FILENAME})")
    p.add_argument("--force", action="store_true", help="Overwrite an existing config")
    p.set_defaults(func=_init)
