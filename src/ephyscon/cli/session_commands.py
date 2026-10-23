"""Session-level commands: ``ephyscon inspect``, ``ephyscon init`` and ``ephyscon sync``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ephyscon.session import Run
from ephyscon.session_config import (
    SESSION_CONFIG_FILENAME,
    SessionConfig,
    default_session_config,
    load_session_config,
    save_session_config,
)
from ephyscon.io.base import open_session
from ephyscon.sync import SyncEdgeCache
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

    p = sub.add_parser("init", help=f"Create {SESSION_CONFIG_FILENAME} for a session")
    p.add_argument("session", type=Path)
    p.add_argument("--format", help="Input format (default: auto-detect)")
    p.add_argument("--excel", help="Legacy lab Excel workbook to migrate run/probe info from")
    p.add_argument("--sheet", help="Sheet name in the workbook (default: session folder name)")
    p.add_argument("-o", "--output", help=f"Output path (default: <session>/{SESSION_CONFIG_FILENAME})")
    p.add_argument("--force", action="store_true", help="Overwrite an existing config")
    p.set_defaults(func=_init)
