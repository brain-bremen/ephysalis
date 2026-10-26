"""Start of a run on the stimulus PC's clock, from the trial stamps of the behavioural record
(VStim ``.tdr`` ``tAbsTrialStart``, time of day in whole seconds, floor): port of
``clocksync.pc_fit`` / ``run_pc_clock`` of the Fritz/Mini BIDS conversions.

The DAQ header gives the start of a DAQ file to 1 s on the DAQ computer's clock, which ran a
constant few minutes ahead of the VStim PC on a day (2004-07-22 335 s, 2005-01-06 398 s,
2005-08-16 885 s). With the stamps, ``PC = offset + (1 + ppm 1e-6) x t`` (``t`` = trial start on
the run's clock): for every candidate rate the trials' 1 s windows are intersected; the rate is
``PRIOR_PPM`` if the data allow it (short runs constrain it poorly), else the one with the widest
common window; the offset is the centre of that window (its width = precision, 1-60 ms). If no
rate gives a common window (stamp jitter, ~0.2 s in some 2004 runs), the windows are widened by
a tolerance. Runs without stamps get the header time minus the median header offset of the
day's runs with a fit (:func:`apply_run_starts`).

``ephyscon timing`` writes ``<work>/run_start/<run>.json`` for runs whose trials carry
``pc_clock_time`` (:mod:`ephyscon.stimulus.vstim_td01`); every step then uses these start
times (``Run.start_time``: NWB / DH5 time axis, BIDS ``acq_time``).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ephyscon.session import Session

logger = logging.getLogger(__name__)

PRIOR_PPM = 35.0  # median DAQ vs PC rate of the well-constrained Fritz runs (>= 100 trials, 2004-2006; IQR ~25-50 ppm)
TOLERANCES_S = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
STAMP_COLUMN = "pc_clock_time"


def pc_fit(c: np.ndarray, s: np.ndarray, ppm_range: float = 300, ppm_step: float = 0.5,
           tolerances: tuple[float, ...] = TOLERANCES_S, prior_ppm: float = PRIOR_PPM) -> dict | None:
    """``PC = offset_s + rate x s`` with ``floor(PC) = c`` (``c``: stamps in s since midnight,
    ``s``: trial starts in s on the run's clock). None with < 3 trials or no common window."""
    if len(c) < 3:
        return None
    ppms = np.arange(-ppm_range, ppm_range + ppm_step / 2, ppm_step)
    for tolerance in tolerances:
        rate = 1 + ppms * 1e-6
        lo = np.max(c[None] - tolerance - rate[:, None] * s[None], axis=1)
        hi = np.min(c[None] + 1 + tolerance - rate[:, None] * s[None], axis=1)
        width = hi - lo
        if width.max() <= 0:
            continue
        feasible = ppms[width > 0]
        k = int(np.argmin(np.abs(ppms - prior_ppm)))
        prior = bool(width[k] > 0)
        if not prior:
            k = int(np.argmax(width))
        return {"ppm": float(ppms[k]), "rate": float(rate[k]), "offset_s": float((lo[k] + hi[k]) / 2),
                "window_ms": float(1000 * width[k]), "n_trials": int(len(c)), "tolerance_s": float(tolerance),
                "ppm_feasible": [float(feasible.min()), float(feasible.max())], "rate_from_prior": prior}
    return None


def trial_stamps(trials: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(stamp in s since midnight, trial start in s) of the trials with a ``pc_clock_time``."""
    if STAMP_COLUMN not in trials:
        return np.zeros(0), np.zeros(0)
    stamps, starts = [], []
    for stamp, start in zip(trials[STAMP_COLUMN], trials.start_time):
        if isinstance(stamp, str) and stamp.count(":") == 2:
            hh, mm, ss = map(int, stamp.split(":"))
            stamps.append(hh * 3600 + mm * 60 + ss)
            starts.append(float(start))
    return np.array(stamps, float), np.array(starts, float)


def fit_method(fit: dict) -> str:
    tolerance = fit.get("tolerance_s", 0)
    feasible = fit["ppm_feasible"]
    return (f"PC clock from {fit['n_trials']} .tdr trial stamps (offset window {fit['window_ms']:.0f} ms"
            + (f", stamp jitter tolerance {tolerance:.1f} s" if tolerance else "")
            + f", DAQ vs PC rate {fit['ppm']:+.1f} ppm"
            + (f" (prior; data allow {feasible[0]:+.0f}..{feasible[1]:+.0f})" if fit.get("rate_from_prior") else "")
            + ")")


def run_start_path(work_dir: Path, run_name: str) -> Path:
    return work_dir / "run_start" / f"{run_name}.json"


def write_run_start(work_dir: Path, run_name: str, header_start: datetime | None, trials: pd.DataFrame) -> dict | None:
    """Fit the PC clock of a run from its trials (:func:`trial_stamps`) and write it; returns the
    record or None (no fit; an old record is removed)."""
    path = run_start_path(work_dir, run_name)
    fit = pc_fit(*trial_stamps(trials)) if header_start is not None else None
    if fit is None:
        path.unlink(missing_ok=True)
        return None
    assert header_start is not None
    midnight = datetime(header_start.year, header_start.month, header_start.day)
    start = midnight + timedelta(seconds=fit["offset_s"])
    record = {"start_time": start.isoformat(), "header_start_time": header_start.isoformat(),
              "header_offset_s": (header_start - start).total_seconds(), "method": fit_method(fit), "fit": fit}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    logger.info("Run %s: start %s on the PC clock (%s)", run_name, start.isoformat(timespec="milliseconds"),
                record["method"])
    return record


def apply_run_starts(session: Session, work_dir: Path) -> None:
    """Set ``start_time`` (and ``extra["start_time_method"]``) of the runs from
    ``<work>/run_start/``; runs without a record but with one on the same day get the header time
    minus the day's median header offset."""
    records: dict[str, dict[str, Any]] = {}
    for run in session.runs:
        path = run_start_path(work_dir, run.name)
        if path.exists():
            records[run.name] = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        return
    offsets: dict[Any, list[float]] = {}
    for record in records.values():
        offsets.setdefault(datetime.fromisoformat(record["header_start_time"]).date(), []).append(record["header_offset_s"])
    for run in session.runs:
        if run.start_time is None:
            continue
        if run.name in records:
            record = records[run.name]
            run.extra["header_start_time"] = run.start_time.isoformat()
            run.start_time = datetime.fromisoformat(record["start_time"])
            run.extra["start_time_method"] = record["method"]
        elif run.start_time.date() in offsets:
            offset = float(np.median(offsets[run.start_time.date()]))
            run.extra["header_start_time"] = run.start_time.isoformat()
            run.start_time -= timedelta(seconds=offset)
            run.extra["start_time_method"] = (f"header start time (1 s) minus the day's DAQ-computer clock offset "
                                              f"{offset:.1f} s from the other runs (accuracy ~2 s; no trial stamps)")
