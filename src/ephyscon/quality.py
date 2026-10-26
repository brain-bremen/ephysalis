"""Channel selection of electrode arrays (``ephyscon quality``, ``[probes.<id>.quality]``): port
of ``channel_quality`` of the Fritz/Mini BIDS conversions (``preprocessing.py``), on the 1 kHz
LFP of each run after mains removal.

From ``n_windows`` windows of ``window_s`` spread evenly over the run (windows in which all
channels are 0 are skipped), per channel:

* ``flat_frac``: fraction of windows with zero variance (bad if > 0);
* ``clip_frac``: fraction of windows with ``|x| >= clip_counts`` (bad if ``>= max_clip_fraction``);
* ``power_z``: robust z (median / 1.4826 MAD over the live channels of the probe) of the median
  log10 power 2-200 Hz (Welch, 500 samples; bad if ``|z| >= max_abs_power_z``);
* ``nb_corr``: median correlation (4th-order Butterworth 2-100 Hz, forward-backward) with the
  mean of the neighbours within ``neighbour_radius_um`` (bad if ``< min_neighbour_correlation``
  or undefined);
* ``line_db``: median 48-52 Hz power over 40-46 and 54-60 Hz, dB (bad if ``>= max_line_db``).

A channel is good if no criterion fails. Writes ``<work>/quality/<run>_<probe>.json``; the CSD
of a run then uses the good channels only (:mod:`ephyscon.csd`), the exports mark the status
(DH5 ``ChannelStatus`` attributes of the LFP blocks, NWB electrodes, BIDS ``channels.tsv``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

    from ephyscon.session import Run
    from ephyscon.session_config import ChannelQualityConfig

logger = logging.getLogger(__name__)

GOOD, BAD = "good", "bad"


def channel_quality(x: Any, xy_um: np.ndarray, config: ChannelQualityConfig, fs: float = 1000.0,
                    valid: list[tuple[int, int]] | None = None) -> list[dict]:
    """Metrics, ``status`` and ``status_description`` per channel of ``x`` (samples x channels,
    counts; array or anything sliceable), ``xy_um`` the positions; ``valid``: sample ranges the
    windows are drawn from (default: all)."""
    from scipy import signal

    n, n_channels = x.shape
    window = int(round(config.window_s * fs))
    valid = valid or [(0, n)]
    spans = [(lo, hi - window) for lo, hi in valid if hi - lo > window]
    total = sum(hi - lo for lo, hi in spans)
    starts = []
    for k in range(config.n_windows):
        position = (k + 0.5) / config.n_windows * total
        for lo, hi in spans:
            if position < hi - lo:
                starts.append(int(lo + position))
                break
            position -= hi - lo
    xy = np.asarray(xy_um, float)
    distance = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    neighbours = [np.flatnonzero((distance[i] > 0) & (distance[i] <= config.neighbour_radius_um))
                  for i in range(n_channels)]
    bandpass = signal.butter(4, (2, 100), btype="band", fs=fs, output="sos")
    clip, flat, log_power, correlation, line = [], [], [], [], []
    for start in starts:
        w = np.asarray(x[start:start + window], dtype=np.float64)
        if not np.any(w):
            continue
        clip.append(np.abs(w).max(0) >= config.clip_counts)
        sd = w.std(0)
        flat.append(sd == 0)
        f, power = signal.welch(w - w.mean(0), fs=fs, nperseg=min(500, len(w)), axis=0)
        log_power.append(np.log10(power[(f >= 2) & (f <= 200)].mean(0) + 1e-30))
        mains = power[(f >= 48) & (f <= 52)].mean(0)
        around = power[((f >= 40) & (f < 46)) | ((f > 54) & (f <= 60))].mean(0)
        line.append(10 * np.log10((mains + 1e-30) / (around + 1e-30)))
        y = signal.sosfiltfilt(bandpass, w, axis=0)
        c = np.full(n_channels, np.nan)
        for i in range(n_channels):
            if len(neighbours[i]) and sd[i] > 0:
                m = y[:, neighbours[i]].mean(1)
                if m.std() > 0:
                    c[i] = np.corrcoef(y[:, i], m)[0, 1]
        correlation.append(c)
    if not log_power:
        raise ValueError("no window with signal for the channel quality")
    clip_a, flat_a, power_a, corr_a, line_a = map(np.array, (clip, flat, log_power, correlation, line))
    median = np.median(power_a, 0)
    live = ~flat_a.all(0)
    mad = 1.4826 * np.median(np.abs(median[live] - np.median(median[live]))) if live.any() else 1.0
    z = (median - np.median(median[live])) / (mad if mad > 0 else 1.0)
    out = []
    for i in range(n_channels):
        r = {"clip_frac": float(clip_a[:, i].mean()), "flat_frac": float(flat_a[:, i].mean()), "power_z": float(z[i]),
             "nb_corr": float(np.nanmedian(corr_a[:, i])) if np.isfinite(corr_a[:, i]).any() else float("nan"),
             "line_db": float(np.median(line_a[:, i])), "n_windows": int(len(power_a))}
        why = []
        if r["flat_frac"] > 0:
            why.append(f"flat (zero variance) in {r['flat_frac']:.0%} of the windows")
        if r["clip_frac"] >= config.max_clip_fraction:
            why.append(f"clipping in {r['clip_frac']:.0%} of the windows")
        if abs(r["power_z"]) >= config.max_abs_power_z:
            why.append(f"power outlier (robust z {r['power_z']:+.1f})")
        if not r["nb_corr"] >= config.min_neighbour_correlation:
            why.append(f"low correlation with the neighbours ({r['nb_corr']:.2f})")
        if r["line_db"] >= config.max_line_db:
            why.append(f"mains 50 Hz {r['line_db']:.0f} dB")
        r["status"] = BAD if why else GOOD
        r["status_description"] = "; ".join(why) if why else (
            f"quality check passed (power z {r['power_z']:+.1f}, neighbour correlation {r['nb_corr']:.2f})")
        out.append(r)
    return out


def quality_path(work_dir: Path, run_name: str, probe_id: str) -> Path:
    return work_dir / "quality" / f"{run_name}_{probe_id}.json"


def recording_quality(lfp: BaseRecording, config: ChannelQualityConfig) -> list[dict]:
    """:func:`channel_quality` of an LFP recording with a probe (whole run)."""
    if not lfp.has_probe():
        raise ValueError("the channel quality needs the probe geometry (geometry in [probes.<id>] of session.toml)")

    class _Traces:  # windows read on demand
        shape = (lfp.get_num_samples(), lfp.get_num_channels())

        def __getitem__(self, index: slice) -> np.ndarray:
            return lfp.get_traces(start_frame=index.start, end_frame=index.stop, return_in_uV=False)

    results = channel_quality(_Traces(), lfp.get_channel_locations()[:, :2], config, fs=lfp.sampling_frequency)
    for channel_id, result in zip(lfp.channel_ids, results):
        result["channel"] = str(channel_id)
    return results


def assess_run(run: Run, probe_id: str, work_dir: Path, config: ChannelQualityConfig) -> list[dict]:
    """Channel quality of one probe in one run from its LFP, to :func:`quality_path`."""
    import spikeinterface.core as si

    from ephyscon.lfp import lfp_folder

    source = lfp_folder(work_dir, run.name, probe_id)
    if not (source / "binary.json").exists():
        raise FileNotFoundError(f"no LFP in {source} (run `ephyscon lfp` first)")
    results = recording_quality(si.load(source), config)
    path = quality_path(work_dir, run.name, probe_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"criteria": config.model_dump(), "channels": results}, indent=1), encoding="utf-8")
    logger.info("%s %s: %d of %d channels good", run.name, probe_id, sum(r["status"] == GOOD for r in results),
                len(results))
    return results


def load_quality(work_dir: Path, run_name: str, probe_id: str) -> dict[str, dict] | None:
    """Channel id -> result of :func:`assess_run`, or None if not assessed."""
    path = quality_path(work_dir, run_name, probe_id)
    if not path.exists():
        return None
    return {r["channel"]: r for r in json.loads(path.read_text(encoding="utf-8"))["channels"]}


def session_status(work_dir: Path, runs: list[Run], probe_id: str) -> dict[str, tuple[str, str]]:
    """Channel id -> (status, description) over the runs of a file: bad if bad in any assessed
    run (description: the reasons per run); empty if no run was assessed."""
    per_run = {run.name: q for run in runs if (q := load_quality(work_dir, run.name, probe_id)) is not None}
    out: dict[str, tuple[str, str]] = {}
    for channel in sorted({c for q in per_run.values() for c in q}):
        bad = {name: q[channel]["status_description"] for name, q in per_run.items()
               if channel in q and q[channel]["status"] == BAD}
        if bad:
            out[channel] = (BAD, "; ".join(f"{name}: {why}" for name, why in bad.items()) if len(per_run) > 1
                            else next(iter(bad.values())))
        else:
            out[channel] = (GOOD, next(q[channel]["status_description"] for q in per_run.values() if channel in q)
                            if len(per_run) == 1 else f"quality check passed in {len(per_run)} runs")
    return out
