"""Reader for DataWave Experimenter's Workbench "UFF" continuous recordings (MS-DOS, 1999; the
Tallon-Baudry et al. 2004 recordings of monkey Eddi), consolidated from ``code/uff/uff.py`` and
``code/bids/eddi.py`` of the Tallon-Baudry BIDS conversion (format reverse-engineered there).

File layout: a file header with the embedded acquisition script, then records back to back,
each a 122-byte header (61 int16 words: word 1 = 0x61, word 2 = timestamp in 0.1 ms ticks,
word 11 = number of channels, words 12... = channel list, word 44 = samples in the record over
all channels, bytes 102-105 = float32 ADC scale in mV per count) and the channel-interleaved
int16 samples (12-bit ADC, 1 kHz per channel).

A recording block was acquired by two PCs, ``M<stem>.UFF`` and ``P<stem>.UFF`` (16 channels
each; the P file may lack its prefix or the M file its extension): one run per block, with
the streams ``M`` (channels ``M0``...``M15``, the reference clock, a true 1 kHz) and ``P``
(``P0``...; its clock runs about 0.4 % slow and has occasional steps). Channel 15 of both
carries the same event pulses (~660 counts, one per stimulus event); the P samples are mapped
onto the M clock through the pulses paired on both (``pair_pulses``; :class:`ClockMap`
interpolates between pairs, so steps are followed). The TTL events of a run are the pulses on
M15 (line 1). Electrodes are defined in ``session.toml`` (probes with ``stream = "M"`` or
``"P"``). Start time: the modification time of the M file (the header has no date).
"""

from __future__ import annotations

import logging
import struct
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from ephyscon.io.base import InputFormat
from ephyscon.session import Run, Session, StreamKind

logger = logging.getLogger(__name__)

HEADER_BYTES = 122
RECORD_MARKER = 0x61
MAGIC = b"UFF"
PULSE_CHANNEL = 15
PULSE_THRESHOLD = 330  # counts; pulses ~660, baseline ~12
NOMINAL_P_RATE = 0.996005  # P samples per M sample (all blocks of Eddi)
SAMPLING_FREQUENCY = 1000.0


def _plausible(buffer: Any, offset: int) -> bool:
    if offset < 0 or offset + HEADER_BYTES > len(buffer):
        return False
    words = struct.unpack_from("<61h", buffer, offset)
    return words[1] == RECORD_MARKER and 0 < words[11] <= 64 and words[44] > 0 and words[44] % words[11] == 0


def _record_size(buffer: Any, offset: int) -> int:
    return HEADER_BYTES + 2 * struct.unpack_from("<h", buffer, offset + 88)[0]


def _first_record(buffer: Any) -> int:
    """Offset of the first record (the one before the 0x61 marker of three plausible records in a row)."""
    data = bytes(buffer[:min(len(buffer), 1 << 20)])
    i = 0
    while True:
        i = data.find(b"\x61\x00", i + 1)
        if i < 0:
            raise ValueError("no UFF data record found")
        offset = i - 2
        if i % 2 == 0 and _plausible(buffer, offset):
            following = offset + _record_size(buffer, offset)
            if _plausible(buffer, following) and _plausible(buffer, following + _record_size(buffer, following)):
                return offset


class UffFile:
    """Records of one UFF file (memory-mapped)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._map: Any = cast(Any, np.memmap(self.path, dtype=np.uint8, mode="r"))  # a buffer for struct
        offset = _first_record(self._map)
        words = struct.unpack_from("<61h", self._map, offset)
        self.n_channels = int(words[11])
        self.channels = [int(c) for c in words[12:12 + self.n_channels]]
        self.adc_scale_mv = float(struct.unpack_from("<f", self._map, offset + 102)[0])
        offsets, ticks, sizes = [], [], []
        while _plausible(self._map, offset):
            words = struct.unpack_from("<61h", self._map, offset)
            if words[11] != self.n_channels:
                raise ValueError(f"{self.path.name}: channel count changes at offset {offset}")
            offsets.append(offset)
            ticks.append(words[2] & 0xFFFF)
            sizes.append(words[44] // self.n_channels)
            offset += HEADER_BYTES + 2 * words[44]
        self.record_offsets = np.array(offsets, dtype=np.int64)
        self.record_samples = np.array(sizes, dtype=np.int64)
        self.record_starts = np.r_[0, np.cumsum(self.record_samples)]
        tick = np.array(ticks, dtype=np.int64)
        self.record_ticks = tick[0] + np.r_[0, np.cumsum(np.diff(tick) % 65536)]  # unwrapped 0.1 ms counter
        self.truncated_bytes = len(self._map) - offset

    @property
    def n_samples(self) -> int:
        return int(self.record_starts[-1])

    def traces(self, start: int, end: int, channels: list[int] | None = None) -> np.ndarray:
        """Samples ``[start, end)`` (int16, samples x channels)."""
        first = int(np.searchsorted(self.record_starts, start, side="right") - 1)
        parts = []
        k = first
        while k < len(self.record_offsets) and self.record_starts[k] < end:
            n = int(self.record_samples[k])
            block = np.frombuffer(self._map, "<i2", n * self.n_channels,
                                  int(self.record_offsets[k]) + HEADER_BYTES).reshape(n, self.n_channels)
            lo = max(start - int(self.record_starts[k]), 0)
            hi = min(end - int(self.record_starts[k]), n)
            parts.append(block[lo:hi])
            k += 1
        out = np.concatenate(parts) if parts else np.zeros((0, self.n_channels), np.int16)
        return out if channels is None else out[:, channels]


@cache
def uff_file(path: str, mtime_ns: int) -> UffFile:  # noqa: ARG001 (mtime: cache key)
    return UffFile(path)


def _open(path: str | Path) -> UffFile:
    return uff_file(str(path), Path(path).stat().st_mtime_ns)


def is_uff(path: Path) -> bool:
    try:
        with open(path, "rb") as file:
            return MAGIC in file.read(16)
    except OSError:
        return False


class UffRecordingSegment(BaseRecordingSegment):
    def __init__(self, path: str, sampling_frequency: float):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._path = path

    def get_num_samples(self) -> int:
        return _open(self._path).n_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        file = _open(self._path)
        start = 0 if start_frame is None else int(start_frame)
        end = file.n_samples if end_frame is None else int(end_frame)
        traces = file.traces(start, end)
        return traces if channel_indices is None else traces[:, channel_indices]


class UffRecordingExtractor(BaseRecording):
    """One UFF file: int16 ADC counts, gains in µV per count at the ADC input."""

    def __init__(self, path: str, prefix: str):
        file = _open(path)
        BaseRecording.__init__(self, SAMPLING_FREQUENCY, [f"{prefix}{c}" for c in file.channels], np.dtype("int16"))
        self.add_recording_segment(UffRecordingSegment(path, SAMPLING_FREQUENCY))
        self.set_channel_gains(file.adc_scale_mv * 1e3)
        self.set_channel_offsets(0.0)
        self._kwargs = {"path": path, "prefix": prefix}


def pulse_onsets(y: np.ndarray, threshold: float = PULSE_THRESHOLD) -> np.ndarray:
    y = np.asarray(y)
    return np.flatnonzero((y[1:] > threshold) & (y[:-1] <= threshold)) + 1


def pulse_offsets(y: np.ndarray, threshold: float = PULSE_THRESHOLD) -> np.ndarray:
    y = np.asarray(y)
    return np.flatnonzero((y[1:] <= threshold) & (y[:-1] > threshold)) + 1


def pair_pulses(pm: np.ndarray, pp: np.ndarray, tol: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Pair the pulse onsets of M and P (a pulse may be missing or doubled on one PC): predict
    the P position of every M pulse from the rate and offset of the last pairs, take the nearest
    P pulse within ``tol`` samples (port of ``eddi.pair_pulses``)."""
    pm, pp = np.asarray(pm, float), np.asarray(pp, float)
    if len(pm) < 2 or len(pp) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rate = NOMINAL_P_RATE

    def hits(offset: float) -> int:
        predicted = rate * pm[:30] + offset
        j = np.clip(np.searchsorted(pp, predicted), 1, len(pp) - 1)
        return int(np.sum(np.minimum(np.abs(pp[j] - predicted), np.abs(pp[j - 1] - predicted)) <= tol))

    offset = max((pp[j] - rate * pm[k] for k in range(min(5, len(pm))) for j in range(min(10, len(pp)))), key=hits)
    pairs: list[tuple[float, float]] = []
    for x in pm:
        predicted = rate * x + offset
        j = int(np.searchsorted(pp, predicted))
        candidates = [c for c in (j - 1, j) if 0 <= c < len(pp)]
        c = min(candidates, key=lambda c: abs(pp[c] - predicted))
        if abs(pp[c] - predicted) <= tol:
            pairs.append((x, pp[c]))
            if len(pairs) >= 20:  # follow rate and offset over the recent pairs
                q = np.array(pairs[-400:])
                rate, offset = np.polyfit(q[:, 0], q[:, 1], 1)
    q = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    _, first = np.unique(q[:, 1], return_index=True)  # a P pulse paired twice (doubled M pulse): the first
    q = q[np.sort(first)]
    return q[:, 0], q[:, 1]


def block_files(folder: Path) -> dict[str, tuple[Path, Path]]:
    """Block stem -> (M file, P file) of the UFF files in ``folder``."""
    files = {p.name.upper(): p for p in folder.iterdir() if p.is_file() and is_uff(p)}

    def stem(name: str) -> str:
        return name[:-4] if name.endswith(".UFF") else name

    blocks = {}
    for name, path in files.items():
        if name.startswith("M"):
            s = stem(name)[1:]
            partner = files.get(f"P{s}.UFF") or files.get(f"P{s}") or files.get(f"{s}.UFF")
            if partner is not None:
                blocks[s] = (path, partner)
            else:
                logger.warning("%s: no P file of block %s", path.name, s)
    return dict(sorted(blocks.items()))


class UffFormat(InputFormat):
    name: ClassVar[str] = "uff"

    def can_read(self, session_path: Path) -> bool:
        return any(p.is_file() and p.suffix.upper() in (".UFF", "") and p.name[:1].upper() == "M" and is_uff(p)
                   for p in session_path.iterdir())

    def open_session(self, session_path: Path) -> Session:
        found = []
        for stem, (m_path, p_path) in block_files(session_path).items():
            try:
                m_file, p_file = _open(m_path), _open(p_path)
            except (ValueError, struct.error) as error:
                logger.warning("block %s: not read (%s)", stem, error)
                continue
            found.append((datetime.fromtimestamp(m_path.stat().st_mtime) - _duration(m_file), stem, m_path, p_path,
                          m_file, p_file))
        runs: list[Run] = []
        for start, stem, m_path, p_path, m_file, p_file in sorted(found, key=lambda item: item[:2]):
            runs.append(Run(
                name=stem, path=session_path, index=len(runs), format=self, start_time=start,
                duration_s=m_file.n_samples / SAMPLING_FREQUENCY, aux_stream="M", streams=["M", "P"],
                extra={"files": {"M": str(m_path), "P": str(p_path)}, "aux_streams": ["M"],
                       "start_time_source": "modification time of the M file minus its duration",
                       "truncated_bytes": {"M": m_file.truncated_bytes, "P": p_file.truncated_bytes}},
            ))
        return Session(name=session_path.name, path=session_path, format=self, runs=runs)

    def default_aux_channels(self, run: Run | None = None):  # type: ignore[no-untyped-def]
        from ephyscon.session_config import AuxChannelMap

        return AuxChannelMap(analog={"eye_x": 13, "eye_y": 14}, digital={"event": 0})

    def get_recording(self, run: Run, kind: StreamKind, probe_id: str | None = None) -> BaseRecording:
        if kind == StreamKind.AUX:
            stream = "M"
        else:
            if probe_id is None:
                raise ValueError(f"probe_id is required for {kind} streams")
            try:
                stream = run.probes[probe_id].streams[kind]
            except KeyError:
                raise KeyError(f"Run {run.name} has no {kind} stream for {probe_id}") from None
        return _extractor(run.extra["files"][stream], stream)

    def aux_analog_recording(self, run: Run) -> BaseRecording:
        """The M channels that belong to no probe (eye position, event pulses)."""
        recording = self.get_recording(run, StreamKind.AUX)
        used = {c for info in run.probes.values() if info.streams.get(StreamKind.AP) == "M"
                for c in (info.channel_ids or recording.channel_ids)}
        return recording.select_channels([c for c in recording.channel_ids if c not in used])

    def _pulses(self, run: Run, stream: str) -> np.ndarray:
        file = _open(run.extra["files"][stream])
        return pulse_onsets(file.traces(0, file.n_samples, [file.channels.index(PULSE_CHANNEL)])[:, 0])

    def clock_maps(self, run: Run, reference: str | None = None):  # type: ignore[no-untyped-def]
        from ephyscon.sync import AUX, ClockMap, RunSync

        pm, pp = pair_pulses(self._pulses(run, "M"), self._pulses(run, "P"))
        maps = {AUX: ClockMap.identity(SAMPLING_FREQUENCY)}
        for probe_id, info in run.probes.items():
            stream = info.streams.get(StreamKind.AP)
            if stream == "M":
                maps[probe_id] = ClockMap.identity(SAMPLING_FREQUENCY)
            elif stream == "P":
                if len(pm) < 2:
                    raise ValueError(f"run {run.name}: fewer than 2 event pulses on both M15 and P15")
                maps[probe_id] = ClockMap(pp.astype(float), pm / SAMPLING_FREQUENCY, SAMPLING_FREQUENCY)
        return RunSync(reference=reference or AUX, maps=maps)

    def read_aux_analog(self, run: Run, index: int) -> np.ndarray:
        file = _open(run.extra["files"]["M"])
        return file.traces(0, file.n_samples, [file.channels.index(index)])[:, 0]

    def aux_analog_gain_v(self, run: Run, index: int) -> float:
        return _open(run.extra["files"]["M"]).adc_scale_mv * 1e-3

    def read_ttl_events(self, run: Run) -> pd.DataFrame:
        """The event pulses on M15 as line 1."""
        y = self.read_aux_analog(run, PULSE_CHANNEL)
        on, off = pulse_onsets(y), pulse_offsets(y)
        events = pd.DataFrame({"aux_sample": np.r_[on, off].astype(np.int64), "line": 1,
                               "state": np.r_[np.ones(len(on)), np.zeros(len(off))].astype(np.int64)})
        return events.sort_values(["aux_sample", "state"], kind="stable", ignore_index=True)

    def read_aux_digital(self, run: Run, bit: int) -> np.ndarray:
        if bit != 0:
            return np.zeros(_open(run.extra["files"]["M"]).n_samples, dtype=bool)
        return self.read_aux_analog(run, PULSE_CHANNEL) > PULSE_THRESHOLD

    def read_messages(self, run: Run) -> pd.DataFrame:
        return pd.DataFrame({"aux_sample": np.zeros(0, np.int64), "text": np.zeros(0, str)})


def _duration(file: UffFile):  # type: ignore[no-untyped-def]
    from datetime import timedelta

    return timedelta(seconds=file.n_samples / SAMPLING_FREQUENCY)


@cache
def _extractor(path: str, prefix: str) -> UffRecordingExtractor:
    return UffRecordingExtractor(path, prefix)
