"""Dataset commands: ``ephysalis run``, ``status`` and ``pause`` (:mod:`ephysalis.dataset`)."""

from __future__ import annotations

import argparse
from pathlib import Path


def _comma_separated(text: str | None) -> list[str] | None:
    return [item for item in text.split(",") if item] if text else None


def _run(args: argparse.Namespace) -> int:
    from ephysalis.dataset.config import load_dataset
    from ephysalis.dataset.runner import Runner, install_signal_handlers

    dataset = load_dataset(args.dataset)
    runner = Runner(dataset, steps=_comma_separated(args.steps), ignore_code_changes=args.ignore_code_changes,
                    dry_run=args.dry_run)
    install_signal_handlers(runner)
    results = runner.run(_comma_separated(args.recording_sessions), parallel=args.parallel)
    return 0 if all(r == "done" for r in results.values()) else 1


def _status(args: argparse.Namespace) -> int:
    from ephysalis.dataset.config import load_dataset
    from ephysalis.dataset.runner import status_table

    print(status_table(load_dataset(args.dataset), ignore_code_changes=args.ignore_code_changes,
                       recording_session_ids=_comma_separated(args.recording_sessions)))
    return 0


def _pause(args: argparse.Namespace) -> int:
    from ephysalis.dataset.config import load_dataset
    from ephysalis.dataset.runner import request_pause

    path = request_pause(load_dataset(args.dataset))
    print(f"Pause requested ({path}): the running steps finish, then the runners stop. "
          f"Resume with: ephysalis run {args.dataset}")
    return 0


def add_dataset_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("run", help="Run the steps of all sessions of a dataset.toml that are not done yet "
                       "(resumes after a pause, an interruption or a failure)")
    p.add_argument("dataset", type=Path, help="dataset.toml")
    p.add_argument("--recording-sessions", help="Only these recording sessions (recording_session_id, comma separated)")
    p.add_argument("--steps", help="Only these steps (comma separated; default: the steps of the dataset)")
    p.add_argument("--parallel", type=int, default=1, help="Sessions at a time (default 1)")
    p.add_argument("--ignore-code-changes", action="store_true",
                   help="Steps done with another version of ephysalis or of the plugins count as done")
    p.add_argument("--dry-run", action="store_true", help="Only show what would run")
    p.set_defaults(func=_run)

    p = sub.add_parser("status", help="Show the state of every step of every session of a dataset.toml")
    p.add_argument("dataset", type=Path, help="dataset.toml")
    p.add_argument("--recording-sessions", help="Only these recording sessions (recording_session_id, comma separated)")
    p.add_argument("--ignore-code-changes", action="store_true", help="As for run")
    p.set_defaults(func=_status)

    p = sub.add_parser("pause", help="Let the runners of a dataset.toml finish their running steps and stop")
    p.add_argument("dataset", type=Path, help="dataset.toml")
    p.set_defaults(func=_pause)
