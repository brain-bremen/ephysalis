"""Better run start times than the recording's (e.g. a header time to 1 s on a computer clock
that runs minutes off), provided by the stimulus system from its own clock
(:meth:`ephyscon.stimulus.base.StimulusSystem.run_start`).

``ephyscon timing`` writes ``<work>/run_start/<run>.json`` for every run the stimulus system
gives a start for; every step then uses these start times (:func:`apply_run_starts`:
``Run.start_time``, the NWB / DH5 time axis of a session file, BIDS ``acq_time``). A run without
one, recorded on the same day as runs with one, is shifted by the median difference between the
recorded and the better start times of those runs (the offset of the recording computer's clock).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ephyscon.session import Session

logger = logging.getLogger(__name__)


@dataclass
class RunStart:
    start_time: datetime  # time of the run's first sample
    method: str  # how it was obtained (stored with the outputs)
    details: dict[str, Any] = field(default_factory=dict)


def run_start_path(work_dir: Path, run_name: str) -> Path:
    return work_dir / "run_start" / f"{run_name}.json"


def write_run_start(work_dir: Path, run_name: str, recorded: datetime | None, start: RunStart | None) -> None:
    """Store (or, with ``start`` None, remove) the better start of a run."""
    path = run_start_path(work_dir, run_name)
    if start is None or recorded is None:
        path.unlink(missing_ok=True)
        return
    record = {"start_time": start.start_time.isoformat(), "recorded_start_time": recorded.isoformat(),
              "recorded_offset_s": (recorded - start.start_time).total_seconds(), "method": start.method,
              "details": start.details}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    logger.info("Run %s: start %s (%s)", run_name, start.start_time.isoformat(timespec="milliseconds"), start.method)


def apply_run_starts(session: Session, work_dir: Path) -> None:
    """Set ``start_time``, ``recorded_start_time`` and ``start_time_method`` of the runs from
    ``<work>/run_start/`` (see the module docs)."""
    records: dict[str, dict[str, Any]] = {}
    for run in session.runs:
        path = run_start_path(work_dir, run.name)
        if path.exists():
            records[run.name] = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        return
    offsets: dict[Any, list[float]] = {}
    for record in records.values():
        offsets.setdefault(datetime.fromisoformat(record["recorded_start_time"]).date(), []).append(
            record["recorded_offset_s"])
    for run in session.runs:
        recorded = run.recorded_start_time or run.start_time
        if recorded is None:
            continue
        if run.name in records:
            run.recorded_start_time = recorded
            run.start_time = datetime.fromisoformat(records[run.name]["start_time"])
            run.start_time_method = records[run.name]["method"]
        elif recorded.date() in offsets:
            offset = float(np.median(offsets[recorded.date()]))
            run.recorded_start_time = recorded
            run.start_time = recorded - timedelta(seconds=offset)
            run.start_time_method = (f"recorded start time minus the day's clock offset {offset:.1f} s of the "
                                     "other runs (no better start of its own)")
