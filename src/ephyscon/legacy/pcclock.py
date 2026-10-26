"""Start of a DAQ run on the VStim PC's clock from the trial stamps of the ``.tdr``
(``tAbsTrialStart``, time of day in whole seconds, floor): port of ``clocksync.pc_fit`` of the
Fritz/Mini BIDS conversions. Used by :meth:`ephyscon.legacy.vstim_td01.VStimTD01System.run_start`
(stored and applied by :mod:`ephyscon.runstart`).

The DAQ header gives the start of a DAQ file to 1 s on the DAQ computer's clock, which ran a
constant few minutes ahead of the VStim PC on a day (2004-07-22 335 s, 2005-01-06 398 s,
2005-08-16 885 s). With the stamps, ``PC = offset + (1 + ppm 1e-6) x t`` (``t`` = trial start on
the run's clock): for every candidate rate the trials' 1 s windows are intersected; the rate is
``PRIOR_PPM`` if the data allow it (short runs constrain it poorly), else the one with the widest
common window; the offset is the centre of that window (its width = precision, 1-60 ms). If no
rate gives a common window (stamp jitter, ~0.2 s in some 2004 runs), the windows are widened by
a tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
