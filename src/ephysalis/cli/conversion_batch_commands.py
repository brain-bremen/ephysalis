"""Conversion batch commands: ``ephysalis run``, ``status`` and ``pause`` (:mod:`ephysalis.conversion_batch`)."""

from __future__ import annotations

import argparse
from pathlib import Path


def _comma_separated(text: str | None) -> list[str] | None:
    return [item for item in text.split(",") if item] if text else None


def _run(args: argparse.Namespace) -> int:
    from ephysalis.conversion_batch.config import load_conversion_batch
    from ephysalis.conversion_batch.runner import Runner, install_signal_handlers

    conversion_batch = load_conversion_batch(args.conversion_batch)
    runner = Runner(conversion_batch, steps=_comma_separated(args.steps), ignore_code_changes=args.ignore_code_changes,
                    dry_run=args.dry_run, source_cache=_source_cache_override(args, conversion_batch))
    install_signal_handlers(runner)
    results = runner.run(_comma_separated(args.recording_sessions), parallel=args.parallel)
    return 0 if all(r == "done" for r in results.values()) else 1


def _source_cache_override(args: argparse.Namespace, conversion_batch):
    """``[source_cache]`` of the file with the options of ``run`` (None: no option given)."""
    from dataclasses import replace

    from ephysalis.conversion_batch.source_cache import SourceCacheMode, check_source_cache_settings

    if args.source_cache is None and args.source_cache_dir is None and args.source_cache_max_gb is None:
        return None
    settings = conversion_batch.source_cache
    if args.source_cache is not None and args.source_cache != settings.mode:
        settings = replace(settings, mode=SourceCacheMode(args.source_cache))
    if args.source_cache_dir is not None:
        settings = replace(settings, dir=args.source_cache_dir.resolve())
    if args.source_cache_max_gb is not None:
        settings = replace(settings, max_size_gb=args.source_cache_max_gb)
    return check_source_cache_settings(settings, "ephysalis run")


def _status(args: argparse.Namespace) -> int:
    from ephysalis.conversion_batch.config import load_conversion_batch
    from ephysalis.conversion_batch.runner import status_table

    print(status_table(load_conversion_batch(args.conversion_batch, check_installed_software=False), ignore_code_changes=args.ignore_code_changes,
                       recording_session_ids=_comma_separated(args.recording_sessions)))
    return 0


def _pause(args: argparse.Namespace) -> int:
    from ephysalis.conversion_batch.config import load_conversion_batch
    from ephysalis.conversion_batch.runner import request_pause

    path = request_pause(load_conversion_batch(args.conversion_batch, check_installed_software=False))
    print(f"Pause requested ({path}): the running steps finish, then the runners stop. "
          f"Resume with: ephysalis run {args.conversion_batch}")
    return 0


def add_conversion_batch_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("run", help="Run the steps of all sessions of a conversion_batch.toml that are not done yet "
                       "(resumes after a pause, an interruption or a failure)")
    p.add_argument("conversion_batch", type=Path, help="conversion_batch.toml")
    p.add_argument("--recording-sessions", help="Only these recording sessions (recording_session_id, comma separated)")
    p.add_argument("--steps", help="Only these steps (comma separated; default: the steps of the conversion batch)")
    p.add_argument("--parallel", type=int, default=1, help="Sessions at a time (default 1)")
    p.add_argument("--ignore-code-changes", action="store_true",
                   help="Steps done with another version of ephysalis or of the plugins count as done")
    p.add_argument("--dry-run", action="store_true", help="Only show what would run")
    p.add_argument("--source-cache", choices=["direct", "local"],
                   help="How the steps read the session folders (overrides [source_cache] mode): directly or a copy "
                        "on a local disk (--source-cache-dir)")
    p.add_argument("--source-cache-dir", type=Path, help="Folder of the copies (overrides [source_cache] dir)")
    p.add_argument("--source-cache-max-gb", type=float,
                   help="All copies together at most this many GB (overrides [source_cache] max_size_gb)")
    p.set_defaults(func=_run)

    p = sub.add_parser("status", help="Show the state of every step of every session of a conversion_batch.toml")
    p.add_argument("conversion_batch", type=Path, help="conversion_batch.toml")
    p.add_argument("--recording-sessions", help="Only these recording sessions (recording_session_id, comma separated)")
    p.add_argument("--ignore-code-changes", action="store_true", help="As for run")
    p.set_defaults(func=_status)

    p = sub.add_parser("pause", help="Let the runners of a conversion_batch.toml finish their running steps and stop")
    p.add_argument("conversion_batch", type=Path, help="conversion_batch.toml")
    p.set_defaults(func=_pause)
