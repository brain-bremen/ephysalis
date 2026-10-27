"""Channel gains from a calibration recording: a sine of known amplitude (the lab's 15 Hz,
100 µV generator) on all inputs (``ephysalis calibrate``, ``[calibration]`` in ``session.toml``).

The sine frequency is fitted on the channel with the largest variance (the generator ran at
about 15.4 Hz rather than 15 Hz): least-squares amplitude of ``sin, cos, 1`` on a 0.02 Hz grid
over ``frequency_range_hz``, then on a 0.001 Hz grid around the best one. The amplitude of every
channel at that frequency (in ADC counts) gives its gain, ``amplitude_uv / amplitude``.
Channels whose sine is below ``min_relative_amplitude`` x the probe's median carry no
calibration signal and get NaN or the probe's median gain (``missing``). Port of ``fit_sine``
and ``calibration_from_15hz`` of the Fritz/Mini BIDS conversion (``calibrate`` of the
Eddi/Walter conversion uses the median for missing channels).

The result, ``<work>/calibration.json`` (gains in µV per count per probe channel), replaces the
gains of the probes' recordings in every later step (:func:`apply_calibration`); a session
without its own calibration recording can use another session's file (``[calibration] file``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ephysalis.session import Run, Session, StreamKind

if TYPE_CHECKING:
    from ephysalis.session_config import CalibrationConfig, SessionConfig

logger = logging.getLogger(__name__)

CALIBRATION_FILENAME = "calibration.json"
_FIT_SECONDS = 60.0  # of the calibration run; the sine is stationary


def _amplitude(t: np.ndarray, frequency: float, y: np.ndarray) -> np.ndarray:
    design = np.c_[np.sin(2 * np.pi * frequency * t), np.cos(2 * np.pi * frequency * t), np.ones_like(t)]
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return np.hypot(coefficients[0], coefficients[1])


def fit_sine(x: np.ndarray, sampling_frequency: float, frequency_range_hz: tuple[float, float] = (14.5, 16.5)
             ) -> tuple[float, np.ndarray]:
    """Frequency (Hz) and amplitude per channel (in the units of ``x``) of a sine in
    ``x`` (samples x channels)."""
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(len(x)) / sampling_frequency
    reference = x[:, int(np.argmax(x.std(axis=0)))]
    grid = np.arange(frequency_range_hz[0], frequency_range_hz[1], 0.02)
    frequency = float(grid[int(np.argmax([_amplitude(t, float(f), reference) for f in grid]))])
    grid = np.arange(frequency - 0.02, frequency + 0.02, 0.001)
    frequency = float(grid[int(np.argmax([_amplitude(t, float(f), reference) for f in grid]))])
    return float(frequency), _amplitude(t, frequency, x)


def probe_calibration(run: Run, probe_id: str, config: CalibrationConfig) -> dict[str, Any]:
    """Gains (µV per count) of a probe's channels from calibration run ``run``."""
    recording = run.get_recording(StreamKind.AP, probe_id)
    n = min(recording.get_num_samples(), int(_FIT_SECONDS * recording.sampling_frequency))
    counts = recording.get_traces(end_frame=n, return_in_uV=False)
    frequency, amplitude = fit_sine(counts, recording.sampling_frequency, config.frequency_range_hz)
    median = float(np.median(amplitude))
    present = amplitude > config.min_relative_amplitude * median
    gains = np.full(amplitude.size, np.nan)
    gains[present] = config.amplitude_uv / amplitude[present]
    if config.missing == "median" and present.any():
        gains[~present] = float(np.median(gains[present]))
    elif config.missing != "nan":
        raise ValueError(f"[calibration] missing must be 'nan' or 'median', not {config.missing!r}")
    return {
        "run": run.name, "frequency_hz": frequency, "amplitude_uv": config.amplitude_uv,
        "channels": [str(c) for c in recording.channel_ids],
        "gains_uv": [None if not np.isfinite(g) else float(g) for g in gains],
        "sine_amplitude_counts": [float(a) for a in amplitude],
        "uncalibrated_gains_uv": [float(g) for g in recording.get_channel_gains()],
        "without_signal": [str(c) for c, ok in zip(recording.channel_ids, present) if not ok],
    }


def compute_calibration(session: Session, config: SessionConfig) -> dict[str, Any]:
    """Calibration of all probes from the run ``[calibration] run`` of the session."""
    name = config.calibration.run
    if name is None:
        raise ValueError("[calibration] run is not set in session.toml")
    run = session.run(name)
    return {"probes": {probe_id: probe_calibration(run, probe_id, config.calibration) for probe_id in run.probes}}


def calibration_path(work_dir: Path) -> Path:
    return Path(work_dir) / CALIBRATION_FILENAME


def load_calibration(config: SessionConfig | None, config_dir: Path | None, work_dir: Path) -> dict[str, Any] | None:
    """The calibration of a session: ``[calibration] file`` (another session's), else the
    work dir's ``calibration.json`` if ``[calibration] run`` is set; None without calibration."""
    if config is None or (config.calibration.run is None and config.calibration.file is None):
        return None
    if config.calibration.file is not None:
        path = Path(config.calibration.file)
        if not path.is_absolute() and config_dir is not None:
            path = config_dir / path
    else:
        path = calibration_path(work_dir)
    if not path.exists():
        raise FileNotFoundError(f"calibration {path} not found (run `ephysalis calibrate` first)")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_calibration(session: Session, calibration: dict[str, Any] | None) -> Session:
    """Put the calibration on the probes of every run: factors on the recorded gains
    (``ProbeInfo.gain_factors``), so that it also applies to a probe's LF band."""
    from dataclasses import replace

    if calibration is None:
        return session
    for probe_id, entry in calibration["probes"].items():
        factors = {c: np.nan if g is None else g / u
                   for c, g, u in zip(entry["channels"], entry["gains_uv"], entry["uncalibrated_gains_uv"])}
        for run in session.runs:
            info = run.probes.get(probe_id)
            if info is None:
                continue
            channel_ids = [str(c) for c in run.get_recording(StreamKind.AP, probe_id).channel_ids]
            missing = [c for c in channel_ids if c not in factors]
            if missing:
                raise ValueError(f"run {run.name}, probe {probe_id}: no calibration for channels {missing}")
            run.probes[probe_id] = replace(info, gain_factors=tuple(factors[c] for c in channel_ids))
    return session


def write_calibration(calibration: dict[str, Any], work_dir: Path) -> Path:
    path = calibration_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=1), encoding="utf-8")
    return path
