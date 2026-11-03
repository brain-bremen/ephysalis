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
                    dry_run=args.dry_run)
    install_signal_handlers(runner)
    results = runner.run(_comma_separated(args.recording_sessions), parallel=args.parallel)
    return 0 if all(r == "done" for r in results.values()) else 1


def _status(args: argparse.Namespace) -> int:
    from ephysalis.conversion_batch.config import load_conversion_batch
    from ephysalis.conversion_batch.runner import status_table

    print(status_table(load_conversion_batch(args.conversion_batch), ignore_code_changes=args.ignore_code_changes,
                       recording_session_ids=_comma_separated(args.recording_sessions)))
    return 0


def _pause(args: argparse.Namespace) -> int:
    from ephysalis.conversion_batch.config import load_conversion_batch
    from ephysalis.conversion_batch.runner import request_pause

    path = request_pause(load_conversion_batch(args.conversion_batch))
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
    p.set_defaults(func=_run)

    p = sub.add_parser("status", help="Show the state of every step of every session of a conversion_batch.toml")
    p.add_argument("conversion_batch", type=Path, help="conversion_batch.toml")
    p.add_argument("--recording-sessions", help="Only these recording sessions (recording_session_id, comma separated)")
    p.add_argument("--ignore-code-changes", action="store_true", help="As for run")
    p.set_defaults(func=_status)

    p = sub.add_parser("pause", help="Let the runners of a conversion_batch.toml finish their running steps and stop")
    p.add_argument("conversion_batch", type=Path, help="conversion_batch.toml")
    p.set_defaults(func=_pause)
