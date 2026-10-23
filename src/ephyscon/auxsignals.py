"""Decoding of auxiliary signals: photodiode trial pulses, running wheel, lasers.

The algorithms port the lab's MATLAB code. They work on the auxiliary stream's own clock;
results are mapped to the reference clock with :mod:`ephyscon.sync`.

(The module is not called ``aux.py`` because ``aux`` is a reserved file name on Windows.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ephyscon.session import Run
    from ephyscon.session_config import AuxChannelMap

logger = logging.getLogger(__name__)

# fillInSampleIndsInStim_CR.m
PHOTODIODE_THRESHOLD = 4000  # ADC counts of the smoothed photodiode signal
# pulses longer than two 16 ms frames on average: the photodiode square covers all stimulus
# frames ("newest Master version"); otherwise a single-frame pulse marks the trial onset
_FRAME_PULSE_MAX_S = 2 * 16e-3
# get_wheelspeed.m
WHEEL_PULSES_PER_REV = 900
WHEEL_RADIUS_CM = 8.0
_WHEEL_KERNEL_LEN = 1000  # 1 ms bins
_WHEEL_KERNEL_ALPHA = 23.5 / 2


def smooth_boxcar(x: np.ndarray, half_width: int, passes: int = 3) -> np.ndarray:
    """Repeated centred moving average with edge values held (port of ``quickFilter_c6``).

    Each pass averages over ``2 * half_width + 1`` samples; three passes approximate a
    Gaussian. The MATLAB code used ``half_width=30`` at 30 kHz (1 ms).
    """
    y = np.asarray(x, dtype=float)
    if half_width < 1 or y.size <= 2 * half_width:
        return y.copy()
    width = 2 * half_width + 1
    for _ in range(passes):
        cs = np.concatenate([[0.0], np.cumsum(y)])
        out = np.empty_like(y)
        out[half_width : y.size - half_width] = (cs[width:] - cs[:-width]) / width
        out[:half_width] = y[0]
        out[y.size - half_width :] = y[-1]
        y = out
    return y


def threshold_crossings(x: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Fractional sample positions where ``x`` crosses ``threshold`` upwards / downwards.

    The position is linearly interpolated between the last sample on one side and the first
    sample on the other side (``ceil`` gives the first sample above/below).
    """
    above = x > threshold
    change = np.flatnonzero(above[1:] != above[:-1])
    frac = (threshold - x[change]) / (x[change + 1] - x[change])
    positions = change + frac
    rising = above[change + 1]
    return positions[rising], positions[~rising]


@dataclass
class PhotodiodePulses:
    onsets: np.ndarray  # fractional aux samples
    offsets: np.ndarray  # fractional aux samples, NaN in single-frame-pulse mode
    mode: str  # "square" (pulse covers the stimulus) or "frame" (single-frame onset pulse)


def detect_photodiode_pulses(
    photodiode: np.ndarray,
    sampling_frequency: float,
    invalid: np.ndarray | None = None,
    threshold: float = PHOTODIODE_THRESHOLD,
    n_trials: int | None = None,
) -> PhotodiodePulses:
    """Trial pulses of the screen photodiode (port of ``extractParadigmStartAndNumberFromNidq``
    and ``fillInSampleIndsInStim_CR`` for runs with a single paradigm).

    ``photodiode`` is in raw ADC counts. ``invalid`` is the "photodiode invalid" digital line:
    the signal is ignored until it first goes low (short dips later on are ignored, as in
    MATLAB). In single-frame-pulse mode ``n_trials`` is needed to place the pulses of the two
    segments before/after a dropout of the photodiode baseline.
    """
    half_width = max(int(round(30 * sampling_frequency / 30000)), 1)
    trial_info = smooth_boxcar(photodiode, half_width)
    if invalid is not None and invalid.size and invalid[0]:
        falling = np.flatnonzero(invalid[:-1] & ~invalid[1:])
        valid_from = falling[0] + 2 if falling.size else trial_info.size
        trial_info[:valid_from] = 0

    onsets, offsets = threshold_crossings(trial_info, threshold)
    if onsets.size and offsets.size:
        # drop an offset before the first onset and an onset without offset at the end
        offsets = offsets[offsets > onsets[0]]
        onsets = onsets[: offsets.size] if onsets.size > offsets.size else onsets
    if onsets.size and offsets.size == onsets.size:
        if np.mean(offsets - onsets) / sampling_frequency > _FRAME_PULSE_MAX_S:
            return PhotodiodePulses(onsets, offsets, "square")

    return _frame_pulses(trial_info, n_trials)


def _frame_pulses(trial_info: np.ndarray, n_trials: int | None) -> PhotodiodePulses:
    # single-frame onset pulses, detected at half maximum. The signal can drop to baseline in
    # between (e.g. screen off): pulses before the first and after the last dropout are
    # placed at the start and end of the trial list, trials in between stay NaN.
    base_rise, base_fall = threshold_crossings(trial_info, 100)
    if base_fall.size == 0 or base_rise.size == 0:
        onsets, _ = threshold_crossings(trial_info, trial_info.max() / 2)
        return PhotodiodePulses(onsets, np.full(onsets.size, np.nan), "frame")
    first = trial_info.copy()
    first[int(np.ceil(base_fall[0])) - 2 :] = 0
    second = trial_info.copy()
    second[: int(np.ceil(base_rise[-1])) + 3] = 0
    onsets1, _ = threshold_crossings(first, first.max() / 2)
    onsets2, _ = threshold_crossings(second, second.max() / 2)
    n = n_trials if n_trials is not None else onsets1.size + onsets2.size
    onsets = np.full(n, np.nan)
    onsets[: min(onsets1.size, n)] = onsets1[:n]
    if onsets2.size:
        onsets[n - min(onsets2.size, n) :] = onsets2[-n:]
    return PhotodiodePulses(onsets, np.full(n, np.nan), "frame")


def wheel_speed(
    rising_edge_times: np.ndarray,
    duration_s: float,
    pulses_per_rev: int = WHEEL_PULSES_PER_REV,
    radius_cm: float = WHEEL_RADIUS_CM,
) -> np.ndarray:
    """Running speed in cm/s at 1 kHz from encoder pulse times (port of ``get_wheelspeed.m``).

    ``rising_edge_times`` are in s on the reference clock; sample ``k`` of the result covers
    ``(k, k + 1]`` ms. The position (pulse count) is differentiated with a derivative of a
    Gaussian (σ ≈ 42.5 ms).
    """
    n_bins = int(np.floor(duration_s * 1000))
    bins = np.ceil(np.asarray(rising_edge_times) * 1000 - 1e-9).astype(np.int64) - 1
    counts = np.bincount(bins[(bins >= 0) & (bins < n_bins)], minlength=n_bins).astype(float)
    position = np.cumsum(counts)

    length = _WHEEL_KERNEL_LEN
    n = np.arange(length) - (length - 1) / 2
    window = np.exp(-0.5 * (_WHEEL_KERNEL_ALPHA * n / ((length - 1) / 2)) ** 2)
    window /= window.sum()
    d_window = np.concatenate([[0.0], np.diff(window)]) * 1000  # per s

    padded = np.concatenate([position, np.full(length, position[-1] if position.size else 0.0)])
    full = np.convolve(padded, d_window)
    start = length // 2  # MATLAB conv(..., 'same')
    velocity = full[start : start + padded.size][: position.size]
    return velocity * 2 * np.pi * radius_cm / pulses_per_rev


class AuxSignals:
    """Access to the roles of a run's auxiliary channels (see :class:`AuxChannelMap`)."""

    def __init__(self, run: Run, channels: AuxChannelMap):
        if run.aux_stream is None:
            raise KeyError(f"Run {run.name} has no auxiliary stream")
        self.run = run
        self.channels = channels
        self.sampling_frequency = run.format.aux_sampling_frequency(run)

    def analog(self, role: str) -> np.ndarray:
        """Raw ADC counts (int16) of an analog role."""
        return self.run.format.read_aux_analog(self.run, self.channels.analog[role])

    def analog_volts(self, role: str) -> np.ndarray:
        return self.analog(role) * self.run.format.aux_analog_gain_v(self.run, self.channels.analog[role])

    def digital(self, role: str) -> np.ndarray:
        return self.run.format.read_aux_digital(self.run, self.channels.digital[role])

    def has(self, role: str) -> bool:
        return role in self.channels.analog or role in self.channels.digital

    def photodiode_pulses(self, n_trials: int | None = None) -> PhotodiodePulses:
        invalid = self.digital("photodiode_invalid") if "photodiode_invalid" in self.channels.digital else None
        return detect_photodiode_pulses(
            self.analog("photodiode").astype(float), self.sampling_frequency, invalid, n_trials=n_trials
        )

    def wheel_rising_edges(self) -> np.ndarray:
        """Aux sample indices of the rising edges of the wheel encoder line."""
        pulses = self.digital("wheel")
        return np.flatnonzero(~pulses[:-1] & pulses[1:]) + 1
