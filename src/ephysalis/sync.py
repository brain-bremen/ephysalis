"""Clock synchronisation between the streams of a run.

Every stream of a run (each probe, the auxiliary NIDQ/ADC stream) records the same sync
pulse train on its own sample clock. Matching the sync edges gives a piecewise-linear map
from each stream's samples to the time base of a *reference* stream (the first probe by
default), which absorbs both start offsets and clock drift.

The MATLAB pipeline resampled the whole NIDQ stream to 30 kHz and shifted it by the smoothed
edge differences. Mapping event times is equivalent for edges and pulses, and much cheaper.

Reading a probe's sync channel means reading its whole AP file, so the edges are extracted
once and cached per run (:class:`SyncEdgeCache`, ``ephysalis sync``). Code that reads the AP
data anyway (e.g. spike sorting) can feed its chunks to an :class:`EdgeAccumulator` and
store the result with :meth:`SyncEdgeCache.store`.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

    from ephysalis.session import Run
    from ephysalis.session_config import AuxChannelMap

logger = logging.getLogger(__name__)

AUX = "aux"  # stream key of the auxiliary stream in RunSync
_CACHE_VERSION = 1


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


@dataclass(frozen=True)
class SyncSource:
    """Where a stream records the sync pulses: one bit of one channel of a recording."""

    recording: BaseRecording
    channel_id: str
    bit: int

    def key(self) -> dict:
        """Identifies the extracted edges; cached edges are only reused if this matches."""
        return {
            "channel_id": str(self.channel_id),
            "bit": self.bit,
            "num_samples": int(self.recording.get_num_samples()),
            "sampling_frequency": float(self.recording.sampling_frequency),
        }


class EdgeAccumulator:
    """Collects the edges of one bit from consecutive chunks of a digital channel."""

    def __init__(self, bit: int, sampling_frequency: float):
        self.bit = bit
        self.sampling_frequency = sampling_frequency
        self.num_samples = 0
        self._samples: list[np.ndarray] = []
        self._rising: list[np.ndarray] = []
        self._previous: bool | None = None
        self._initial = False

    def add(self, word: np.ndarray) -> None:
        """Add the next chunk (1-d) of the digital channel."""
        bits = ((np.asarray(word).astype(np.int64) >> self.bit) & 1).astype(bool)
        if bits.size == 0:
            return
        n_new = bits.size
        offset = self.num_samples
        if self._previous is None:
            self._initial = bool(bits[0])
        else:
            # prepend the last sample of the previous chunk to catch an edge at the boundary
            bits = np.concatenate([[self._previous], bits])
            offset -= 1
        edges = binary_edges(bits, self.sampling_frequency, offset=offset)
        self._samples.append(edges.samples)
        self._rising.append(edges.rising)
        self._previous = bool(bits[-1])
        self.num_samples += n_new

    def edges(self) -> Edges:
        return Edges(
            samples=np.concatenate(self._samples) if self._samples else np.zeros(0, np.int64),
            rising=np.concatenate(self._rising) if self._rising else np.zeros(0, bool),
            sampling_frequency=self.sampling_frequency,
            initial_level=self._initial,
        )


def digital_edges(
    recording: BaseRecording,
    channel: int | str,
    bit: int,
    chunk_s: float = 60.0,
) -> Edges:
    """Edges of one bit of a digital channel, read in chunks (works on multi-GB AP files)."""
    channel_id = recording.channel_ids[channel] if isinstance(channel, int) else channel
    accumulator = EdgeAccumulator(bit, recording.sampling_frequency)
    n = recording.get_num_samples()
    chunk = max(int(chunk_s * recording.sampling_frequency), 2)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        accumulator.add(recording.get_traces(start_frame=start, end_frame=stop, channel_ids=[channel_id])[:, 0])
    return accumulator.edges()


class SyncEdgeCache:
    """Sync edges of all streams of a run, stored in ``<directory>/<run name>.npz``."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def path(self, run_name: str) -> Path:
        return self.directory / f"{run_name}.npz"

    def load(self, run_name: str) -> dict[str, tuple[dict, Edges]]:
        """``{stream: (source key, edges)}``; empty if nothing (compatible) is cached."""
        path = self.path(run_name)
        if not path.exists():
            return {}
        with np.load(path) as data:
            info = json.loads(str(data["info"]))
            if info.get("version") != _CACHE_VERSION:
                return {}
            return {
                stream: (
                    entry["key"],
                    Edges(
                        samples=data[f"{stream}.samples"],
                        rising=data[f"{stream}.rising"],
                        sampling_frequency=entry["key"]["sampling_frequency"],
                        initial_level=entry["initial_level"],
                    ),
                )
                for stream, entry in info["streams"].items()
            }

    def save(self, run_name: str, entries: dict[str, tuple[dict, Edges]]) -> None:
        info = {
            "version": _CACHE_VERSION,
            "streams": {
                stream: {"key": key, "initial_level": edges.initial_level}
                for stream, (key, edges) in entries.items()
            },
        }
        arrays: dict[str, Any] = {"info": np.array(json.dumps(info))}
        for stream, (_, edges) in entries.items():
            arrays[f"{stream}.samples"] = edges.samples
            arrays[f"{stream}.rising"] = edges.rising
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / f"{run_name}.tmp.npz"
        np.savez(tmp, **arrays)
        tmp.replace(self.path(run_name))

    def store(self, run_name: str, stream: str, source: SyncSource, edges: Edges) -> None:
        """Add or replace the edges of one stream (e.g. collected during spike sorting)."""
        entries = self.load(run_name)
        entries[stream] = (source.key(), edges)
        self.save(run_name, entries)


def sync_sources(run: Run, aux_channels: AuxChannelMap | None = None) -> dict[str, SyncSource]:
    """Sync source of every probe of a run and of its aux stream (key :data:`AUX`)."""
    sources = {pid: run.format.sync_source(run, pid, aux_channels) for pid in run.probes}
    if run.aux_stream is not None:
        sources[AUX] = run.format.sync_source(run, None, aux_channels)
    return sources


def load_or_extract_edges(
    run: Run,
    aux_channels: AuxChannelMap | None = None,
    cache: SyncEdgeCache | None = None,
    max_workers: int = 4,
) -> dict[str, Edges]:
    """Sync edges of all streams of a run: from ``cache`` where valid, otherwise extracted
    (streams in parallel, reading is I/O-bound) and added to ``cache``."""
    sources = sync_sources(run, aux_channels)
    cached = cache.load(run.name) if cache is not None else {}
    result: dict[str, Edges] = {}
    missing = []
    for stream, source in sources.items():
        if stream in cached and cached[stream][0] == source.key():
            result[stream] = cached[stream][1]
        else:
            missing.append(stream)
    if not missing:
        return result

    logger.info("Run %s: extracting sync edges of %s", run.name, ", ".join(missing))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            stream: pool.submit(digital_edges, sources[stream].recording, sources[stream].channel_id, sources[stream].bit)
            for stream in missing
        }
        for stream, future in futures.items():
            result[stream] = future.result()
    if cache is not None:
        cache.save(run.name, {stream: (sources[stream].key(), result[stream]) for stream in sources})
    return result


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
    extrapolated with a straight line fitted to all edges. ``rate`` (samples per reference
    second) overrides the slope of that line where the format knows the true rate better, e.g. a
    clock with steps (UFF "P": one rate, an offset per stretch).
    """

    source_samples: np.ndarray
    reference_times: np.ndarray
    sampling_frequency: float  # nominal rate of the source stream
    rate: float | None = None
    _fit: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.source_samples.size >= 2:
            slope, intercept = np.polyfit(self.source_samples, self.reference_times, 1)
        elif self.source_samples.size == 1:
            slope = 1 / self.sampling_frequency
            intercept = self.reference_times[0] - slope * self.source_samples[0]
        else:
            slope, intercept = 1 / self.sampling_frequency, 0.0
        if self.rate is not None and self.source_samples.size:
            slope = 1 / self.rate
            intercept = float(np.mean(self.reference_times - slope * self.source_samples))
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
    cache: SyncEdgeCache | None = None,
) -> RunSync:
    """Build clock maps for all probes and the aux stream of a run from their sync edges
    (read from ``cache`` if it holds valid edges for the run), or take them from the format
    if it has synchronised timestamps (:meth:`InputFormat.clock_maps`)."""
    from_format = run.format.clock_maps(run, reference)
    if from_format is not None:
        return from_format
    probe_ids = list(run.probes)
    if not probe_ids:
        raise ValueError(f"Run {run.name} has no probes")
    reference = reference or probe_ids[0]

    edges = load_or_extract_edges(run, aux_channels, cache)

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
