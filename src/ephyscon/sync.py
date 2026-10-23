"""Clock synchronisation between the streams of a run.

Every stream of a run (each probe, the auxiliary NIDQ/ADC stream) records the same sync
pulse train on its own sample clock. Matching the sync edges gives a piecewise-linear map
from each stream's samples to the time base of a *reference* stream (the first probe by
default), which absorbs both start offsets and clock drift.

The MATLAB pipeline resampled the whole NIDQ stream to 30 kHz and shifted it by the smoothed
edge differences. Mapping event times is equivalent for edges and pulses, and much cheaper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

    from ephyscon.session import Run
    from ephyscon.session_config import AuxChannelMap

logger = logging.getLogger(__name__)

AUX = "aux"  # stream key of the auxiliary stream in RunSync


@dataclass
class Edges:
    """Transitions of a binary signal: ``samples[i]`` is the first sample of the new level."""

    samples: np.ndarray  # int64
    rising: np.ndarray  # bool, True for low -> high
    sampling_frequency: float
    initial_level: bool = False

    @property
    def times(self) -> np.ndarray:
        """Nominal times in s (sample index / nominal sampling rate)."""
        return self.samples / self.sampling_frequency


def binary_edges(signal: np.ndarray, sampling_frequency: float, offset: int = 0) -> Edges:
    """Edges of a boolean signal (sample index of the first sample at the new level)."""
    signal = np.asarray(signal, dtype=bool)
    change = np.flatnonzero(signal[1:] != signal[:-1]) + 1
    return Edges(
        samples=change.astype(np.int64) + offset,
        rising=signal[change],
        sampling_frequency=sampling_frequency,
        initial_level=bool(signal[0]) if signal.size else False,
    )


def digital_edges(
    recording: BaseRecording,
    channel: int | str,
    bit: int,
    chunk_s: float = 60.0,
) -> Edges:
    """Edges of one bit of a digital channel, read in chunks (works on multi-GB AP files)."""
    channel_id = recording.channel_ids[channel] if isinstance(channel, int) else channel
    fs = recording.sampling_frequency
    n = recording.get_num_samples()
    chunk = max(int(chunk_s * fs), 2)
    samples, rising = [], []
    previous: bool | None = None
    initial = False
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        word = recording.get_traces(start_frame=start, end_frame=stop, channel_ids=[channel_id])[:, 0]
        bits = ((word.astype(np.int64) >> bit) & 1).astype(bool)
        if previous is None:
            initial = bool(bits[0])
        else:
            bits = np.concatenate([[previous], bits])
        edges = binary_edges(bits, fs, offset=start if previous is None else start - 1)
        samples.append(edges.samples)
        rising.append(edges.rising)
        previous = bool(bits[-1])
    return Edges(
        samples=np.concatenate(samples) if samples else np.zeros(0, np.int64),
        rising=np.concatenate(rising) if rising else np.zeros(0, bool),
        sampling_frequency=fs,
        initial_level=initial,
    )


def match_edges(source: Edges, target: Edges, tolerance_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Pair sync edges of two streams; returns the paired (source, target) sample indices.

    Edges are paired by polarity and nominal time after removing the median start offset,
    so streams may start at different times and have slightly different clocks. The offset
    between the streams must be less than half a sync period (SpikeGLX starts all streams
    together, so it is a few ms at most).
    """
    pairs_src, pairs_dst = [], []
    for polarity in (True, False):
        s = source.samples[source.rising == polarity]
        t = target.samples[target.rising == polarity]
        if s.size == 0 or t.size == 0:
            continue
        ts, tt = s / source.sampling_frequency, t / target.sampling_frequency
        nearest = np.clip(np.searchsorted(tt, ts), 1, tt.size - 1) if tt.size > 1 else np.zeros(ts.size, int)
        if tt.size > 1:
            left = nearest - 1
            nearest = np.where(np.abs(tt[left] - ts) <= np.abs(tt[nearest] - ts), left, nearest)
        pairs_src.append(s)
        pairs_dst.append(t[nearest])
    if not pairs_src:
        raise ValueError("No sync edges to match")
    src = np.concatenate(pairs_src)
    dst = np.concatenate(pairs_dst)
    delta = dst / target.sampling_frequency - src / source.sampling_frequency
    good = np.abs(delta - np.median(delta)) < tolerance_s
    order = np.argsort(src[good])
    src, dst = src[good][order], dst[good][order]
    # each target edge may be used once
    _, unique = np.unique(dst, return_index=True)
    unique.sort()
    return src[unique], dst[unique]


@dataclass
class ClockMap:
    """Piecewise-linear map from samples of a stream to seconds on the reference clock.

    Between matched sync edges the map is interpolated linearly. Outside them it is
    extrapolated with a straight line fitted to all edges.
    """

    source_samples: np.ndarray
    reference_times: np.ndarray
    sampling_frequency: float  # nominal rate of the source stream
    _fit: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.source_samples.size >= 2:
            slope, intercept = np.polyfit(self.source_samples, self.reference_times, 1)
        elif self.source_samples.size == 1:
            slope = 1 / self.sampling_frequency
            intercept = self.reference_times[0] - slope * self.source_samples[0]
        else:
            slope, intercept = 1 / self.sampling_frequency, 0.0
        self._fit = (float(slope), float(intercept))

    @classmethod
    def identity(cls, sampling_frequency: float) -> ClockMap:
        return cls(np.zeros(0), np.zeros(0), sampling_frequency)

    @property
    def effective_sampling_frequency(self) -> float:
        """Rate of the stream measured in reference-clock seconds."""
        return 1 / self._fit[0]

    def to_reference_time(self, samples: np.ndarray | float) -> np.ndarray:
        samples = np.asarray(samples, dtype=float)
        slope, intercept = self._fit
        times = slope * samples + intercept
        if self.source_samples.size >= 2:
            inside = (samples >= self.source_samples[0]) & (samples <= self.source_samples[-1])
            times = np.where(inside, np.interp(samples, self.source_samples, self.reference_times), times)
        return times

    def to_samples(self, times: np.ndarray | float) -> np.ndarray:
        """Inverse of :meth:`to_reference_time` (fractional sample indices)."""
        times = np.asarray(times, dtype=float)
        slope, intercept = self._fit
        samples = (times - intercept) / slope
        if self.source_samples.size >= 2:
            inside = (times >= self.reference_times[0]) & (times <= self.reference_times[-1])
            samples = np.where(inside, np.interp(times, self.reference_times, self.source_samples), samples)
        return samples

    def max_residual_s(self) -> float:
        """Largest deviation of the edges from the linear fit (clock jitter + drift)."""
        if self.source_samples.size < 2:
            return 0.0
        slope, intercept = self._fit
        return float(np.max(np.abs(slope * self.source_samples + intercept - self.reference_times)))


@dataclass
class RunSync:
    """Clock maps of all streams of a run onto the reference stream's clock."""

    reference: str  # probe id of the reference stream
    maps: dict[str, ClockMap]  # probe ids and AUX

    def to_reference_time(self, stream: str, samples: np.ndarray | float) -> np.ndarray:
        return self.maps[stream].to_reference_time(samples)

    def to_samples(self, stream: str, times: np.ndarray | float) -> np.ndarray:
        return self.maps[stream].to_samples(times)


def synchronize_run(
    run: Run,
    aux_channels: AuxChannelMap | None = None,
    reference: str | None = None,
    min_edges: int = 2,
) -> RunSync:
    """Build clock maps for all probes and the aux stream of a run from their sync edges."""
    probe_ids = list(run.probes)
    if not probe_ids:
        raise ValueError(f"Run {run.name} has no probes")
    reference = reference or probe_ids[0]

    edges = {pid: run.format.sync_edges(run, pid, aux_channels) for pid in probe_ids}
    if run.aux_stream is not None:
        edges[AUX] = run.format.sync_edges(run, None, aux_channels)

    ref_edges = edges[reference]
    maps = {}
    for stream, stream_edges in edges.items():
        if stream == reference:
            src = ref_edges.samples
            dst = src
        else:
            src, dst = match_edges(stream_edges, ref_edges)
        if src.size < min_edges:
            raise ValueError(
                f"Run {run.name}: only {src.size} matching sync edges between {stream} and {reference}"
            )
        clock = ClockMap(src.astype(float), dst / ref_edges.sampling_frequency, stream_edges.sampling_frequency)
        logger.info(
            "Run %s: %s -> %s from %d sync edges, rate %.4f Hz, max residual %.3f ms",
            run.name, stream, reference, src.size, clock.effective_sampling_frequency,
            clock.max_residual_s() * 1e3,
        )
        maps[stream] = clock
    return RunSync(reference=reference, maps=maps)
