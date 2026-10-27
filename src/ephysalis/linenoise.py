"""Mains removal in the LFP with the recorded mains reference: port of the lab's MATLAB
``dh_remove_50hz`` (M. Borisov 2004; ``analyze_50hz.m``, ``process_channel.m``), used on the
Fritz/Mini epidural recordings (``[line_noise]`` in ``session.toml``, applied by ``ephysalis lfp``).

Method (per run, on the 1 kHz LFP):

1. Cycle starts: the upward zero crossings of the recorded mains, sample ``i`` with
   ``ref[i] < 0 <= ref[i + 1]``, on the reference as recorded (``analyze_50hz.m``) or after
   subtracting its mean (``subtract_reference_mean``, as the Fritz/Mini BIDS conversions; with
   it the result equals their files sample by sample). A crossing that moves by one sample
   changes the templates of the following ~4 s, so the two differ visibly.
2. The 30 samples from each cycle start are one fragment. Each of the 30 phase positions is
   low-pass filtered across cycles with the 4th-order IIR of ``process_channel.m`` (one value per
   cycle; the template follows changes of the noise over about 4 s); the output at cycle ``k``
   (including cycle ``k``) is that cycle's template.
3. The first ``min(30, cycle length)`` template values are subtracted from cycle ``k``; the last
   cycle runs to the end of the data. int16 data are rounded as MATLAB's ``int16`` (to nearest,
   ties away from zero, saturating).

``process_channel.m`` indexes past its 30-value template when a cycle is longer than 30
samples; here such cycles get only the 30 values (as in the Python port of the Fritz BIDS
conversion). The reference of a run is used only if it is mains (``reference_quality``; not in
the original, which was run only where the reference was known to be mains): at least
``min_cycle_fraction`` of its cycles last 19-21 ms and its 50 Hz peak lies ``min_peak_db``
above the spectrum 5-20 Hz away. Otherwise the run is left unchanged. The result of each run
is in ``line_noise.json`` of its LFP folder.

The template of cycle ``k`` uses cycles up to ``k`` only (causal across cycles), so the
removal reaches no further into the future than the cycle itself (``docs/filters.md``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal

from ephysalis.session import Run, StreamKind

if TYPE_CHECKING:
    from ephysalis.session_config import LineNoiseConfig, SessionConfig
    from ephysalis.sync import RunSync

logger = logging.getLogger(__name__)

# 4th-order IIR low-pass across cycles, coefficients of process_channel.m
B = np.array([9.5041814898263652e-009, 3.8016725959305461e-008, 5.7025088938958195e-008,
              3.8016725959305461e-008, 9.5041814898263652e-009])
A = np.array([1, -3.9480637269326655, 5.8455356203529982, -3.8468597632865809, 0.9493880219331513])
N_POINTS = 30
RATE_HZ = 1000.0  # dh_remove_50hz requires 1 kHz signals
STATUS_FILENAME = "line_noise.json"
_CHANNEL_BLOCK = 8  # channels processed at once (memory: cycles x 30 x channels float64)


def zero_crossings(ref: np.ndarray) -> np.ndarray:
    """Samples ``i`` with ``ref[i] < 0 <= ref[i + 1]`` (``analyze_50hz.m``, 0-based)."""
    ref = np.asarray(ref)
    return np.flatnonzero((ref[:-1] < 0) & (ref[1:] >= 0))


def line_noise_templates(x: np.ndarray, crossings: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    """What ``process_channel.m`` subtracts from ``x`` (samples x channels) for the cycle
    starts ``crossings``."""
    n = x.shape[0]
    full = crossings + n_points <= n  # cycles with a full fragment
    starts = crossings[full]
    ends = np.r_[crossings[1:], n][full]  # a cycle ends at the next crossing (with or without full fragment)
    lengths = np.minimum(n_points, ends - starts)
    fragments = x[starts[:, None] + np.arange(n_points)[None]]  # cycles x n_points x channels
    templates = signal.lfilter(B, A, fragments, axis=0)
    subtracted = np.zeros_like(x, dtype=np.float64)
    for start, length, template in zip(starts, lengths, templates):
        subtracted[start:start + length] = template[:length]
    return subtracted


def matlab_int16(x: np.ndarray) -> np.ndarray:
    """MATLAB ``int16(x)``: round to nearest with ties away from zero, saturate."""
    return np.clip(np.sign(x) * np.floor(np.abs(x) + 0.5), -32768, 32767).astype(np.int16)


def remove_line_noise(x: np.ndarray, ref: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    """``x`` (samples x channels, or 1-D) without the mains noise locked to ``ref``; int16
    input gives int16 output (rounded as MATLAB), else float64."""
    x = np.asarray(x)
    one_d = x.ndim == 1
    data = x[:, None] if one_d else x
    cleaned = data.astype(np.float64) - line_noise_templates(data.astype(np.float64), zero_crossings(ref), n_points)
    if x.dtype == np.int16:
        cleaned = matlab_int16(cleaned)
    return cleaned[:, 0] if one_d else cleaned


def reference_quality(ref: np.ndarray, sampling_frequency: float = RATE_HZ, max_seconds: float = 300.0
                      ) -> tuple[float, float]:
    """Fraction of the cycles (zero crossings as used) lasting 19-21 ms, and the 50 Hz peak in
    dB above the median spectrum 5-20 Hz away (Welch, 1 s Hann segments)."""
    ref = np.asarray(ref[:int(max_seconds * sampling_frequency)], dtype=np.float64)
    periods_ms = np.diff(zero_crossings(ref)) * 1000.0 / sampling_frequency
    fraction = float(np.mean((periods_ms >= 19) & (periods_ms <= 21))) if periods_ms.size else 0.0
    if ref.size < 2 * sampling_frequency:
        return fraction, float("-inf")
    frequencies, power = signal.welch(ref - ref.mean(), fs=sampling_frequency, window="hann",
                                      nperseg=int(sampling_frequency))
    peak = power[np.abs(frequencies - 50) <= 1].max()
    floor = np.median(power[(np.abs(frequencies - 50) >= 5) & (np.abs(frequencies - 50) <= 20)])
    return fraction, float(10 * np.log10(peak / floor)) if floor > 0 else float("inf")


def reference_on_lfp(run: Run, probe_id: str, n_samples: int, index: int, sync: RunSync) -> np.ndarray:
    """Analog input ``index`` of the aux stream at the 1 kHz LFP samples of a probe: the
    samples themselves if they coincide (same stream and rate), else resampled to 1 kHz and
    interpolated at the LFP sample times (through the run's clock maps)."""
    from ephysalis.sync import AUX

    probe_rate = run.get_recording(StreamKind.AP, probe_id).sampling_frequency
    lfp_probe_samples = np.arange(n_samples) * probe_rate / RATE_HZ
    aux_samples = sync.to_samples(AUX, sync.to_reference_time(probe_id, lfp_probe_samples))
    aux_rate = run.format.aux_sampling_frequency(run)
    raw = run.format.read_aux_analog(run, index).astype(np.float64)
    if aux_rate != RATE_HZ:  # the LFP's anti-alias resampling (ephysalis.lfp)
        import spikeinterface.core as si
        import spikeinterface.preprocessing as spre

        raw = spre.resample(si.NumpyRecording(raw[:, None], aux_rate), RATE_HZ, dtype="float64").get_traces()[:, 0]
    positions = aux_samples * RATE_HZ / aux_rate
    exact = np.round(positions)
    if np.allclose(positions, exact, atol=1e-6) and exact.min() >= 0 and exact.max() < raw.size:
        return raw[exact.astype(np.int64)]
    return np.interp(positions, np.arange(raw.size), raw)


def _lfp_traces(folder: Path) -> np.memmap:
    meta = json.loads((folder / "binary.json").read_text(encoding="utf-8"))["kwargs"]
    n_channels = meta["num_channels"]
    path = folder / "traces_cached_seg0.raw"
    dtype = np.dtype(meta["dtype"])
    n_samples = path.stat().st_size // (dtype.itemsize * n_channels)
    return np.memmap(path, dtype=dtype, mode="r+", shape=(n_samples, n_channels), offset=meta.get("file_offset", 0))


def clean_lfp(run: Run, probe_id: str, folder: Path, config: SessionConfig, sync: RunSync) -> dict[str, Any]:
    """Remove the mains from the LFP of one probe in one run, in place (``folder``: its LFP
    folder); writes and returns the status (``line_noise.json``)."""
    line_noise: LineNoiseConfig = config.line_noise
    status_path = folder / STATUS_FILENAME
    if status_path.exists():
        return json.loads(status_path.read_text(encoding="utf-8"))
    role = line_noise.reference_role
    if role not in config.aux_channels.analog:
        raise ValueError(f"[line_noise] needs the analog role {role!r} in [aux_channels.analog]")
    index = config.aux_channels.analog[role]
    traces = _lfp_traces(folder)
    meta = json.loads((folder / "binary.json").read_text(encoding="utf-8"))["kwargs"]
    if meta["sampling_frequency"] != RATE_HZ:
        raise ValueError(f"mains removal needs a {RATE_HZ:g} Hz LFP, not {meta['sampling_frequency']} Hz")
    ref = reference_on_lfp(run, probe_id, traces.shape[0], index, sync)
    fraction, peak_db = reference_quality(ref)
    status: dict[str, Any] = {"run": run.name, "probe": probe_id, "reference": f"{role} (analog input {index})",
                              "reference_cycle_fraction": fraction, "reference_peak_db": peak_db,
                              "method": "dh_remove_50hz port (ephysalis.linenoise)",
                              "reference_mean_subtracted": line_noise.subtract_reference_mean}
    if fraction < line_noise.min_cycle_fraction or peak_db < line_noise.min_peak_db:
        status["applied"] = False
        status["reason"] = (f"reference is not mains: {fraction:.2f} of the cycles 19-21 ms (min "
                            f"{line_noise.min_cycle_fraction}), 50 Hz peak {peak_db:.1f} dB (min {line_noise.min_peak_db})")
        logger.warning("%s %s: mains removal not applied (%s)", run.name, probe_id, status["reason"])
    else:
        crossings = zero_crossings(ref - ref.mean() if line_noise.subtract_reference_mean else ref)
        for first in range(0, traces.shape[1], _CHANNEL_BLOCK):
            block = np.asarray(traces[:, first:first + _CHANNEL_BLOCK])
            cleaned = block.astype(np.float64) - line_noise_templates(block.astype(np.float64), crossings)
            traces[:, first:first + _CHANNEL_BLOCK] = (matlab_int16(cleaned) if traces.dtype == np.int16
                                                       else cleaned.astype(traces.dtype))
        traces.flush()
        status["applied"] = True
        status["n_cycles"] = int(crossings.size)
    status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
    return status
