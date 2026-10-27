"""Reader for Multi Channel Systems MC_Rack files (``*.mcd``, MC_Rack 3.8-4.0, 2010-2011; the
lab's USB-ME256 recordings of monkey Iris). Reverse-engineered from the files, with the MCS
Neuroshare library (nsMCDLibrary 3.1) and its documentation as reference: samples, channels,
digital words and trigger events equal what the library returns for MC_Rack 3.8/4.0 files
(header versions 10 and 11).

The file format is described in ``docs/legacy/mcd.md``.

MC_Rack writes a long recording into several files (``name.mcd``, ``name0001.mcd``, ... of at
most 2 or 3 GB) that continue each other without gap; consecutive files of a folder whose blocks
continue (and whose streams are the same) form one run, named after the first. Only continuous
recordings are read (not the triggered sweep mode).

Streams are named by their tag (``elec0001`` ...), channel ids are the MC_Rack labels (``01A``,
``61D``, ``A1``, ``D1``). Analog samples are int16 counts (sample - AD zero) with the resolution
as gain (µV per count at the amplifier input); the digital stream (``digi0001``, one 16-bit word)
is the aux stream: its bits 0-15 are TTL lines 1-16. Trigger events with trial synchronisation
(VStim 1.x through the USB box) are the trial descriptors (:meth:`McdFormat.read_trial_descriptors`,
as the ``TD01`` records of DAQ files). Probes are defined in ``session.toml`` (``stream =
"elec0001"``, ``channels``). Digital transitions take a pass over the digital stream and are
cached on disk (:func:`ephysalis.legacy.daq.cache_dir`).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import struct
from dataclasses import dataclass, field
from collections.abc import Collection
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from ephysalis.io.base import InputFormat
from ephysalis.session import Run, Session, StreamKind

if TYPE_CHECKING:
    from ephysalis.session_config import AuxChannelMap
    from ephysalis.sync import RunSync

logger = logging.getLogger(__name__)

MAGIC = b"MCSSTRM "
EPOCH = datetime(1601, 1, 1)
CHANNEL_RECORDS = 804  # STRMHDR: start of the channel records
DATA_STREAMS = ("elec", "anlg", "filt", "chtl", "digi")  # streams with samples
TRIGGER_EVENT_BYTES = (10, 14, 22)  # time + value (+ number, reserved) (+ trial, stimulus)
_CACHE_VERSION = 1


class McdFormatError(ValueError):
    pass


@dataclass
class McdStream:
    tag: str  # e.g. elec0001
    type_name: str  # STRMHDR type: analog, filter, digital, trigger, average, ...
    sampling_frequency: float
    channel_ids: list[str]
    hwids: list[int]
    unit: str
    ad_zero: int
    resolution: float  # unit per count
    samples_per_block: int

    @property
    def kind(self) -> str:
        return self.tag[:4]

    @property
    def is_digital(self) -> bool:
        return self.kind == "digi"

    @property
    def chunk_size(self) -> int:
        """Body size of a data chunk (block times + samples)."""
        return 16 + 2 * self.samples_per_block * len(self.channel_ids)


def _cstring(data: bytes) -> str:
    return data.split(b"\0")[0].decode("latin-1")


def _chunks(data: bytes, start: int = 0) -> list[tuple[str, int, int]]:
    """(tag, body offset, size) of the chunks in ``data``."""
    out, pos = [], start
    while pos + 16 <= len(data):
        tag, size = struct.unpack_from("<8sq", data, pos)
        out.append((tag.decode("latin-1"), pos + 16, size))
        pos += 16 + size
    return out


def parse_streams(header: bytes) -> tuple[tuple[int, ...], list[McdStream]]:
    """MCSHDR version and the streams of a ``LISThdr`` body."""
    version: tuple[int, ...] = ()
    streams: list[McdStream] = []
    pending: tuple[bytes, Any] | None = None
    for tag, start, size in _chunks(header):
        body = header[start:start + size]
        if tag == "MCSHDR  ":
            version = struct.unpack_from("<6H", body)
        elif tag == "STRMHDR ":
            pending = (body, None)
        elif tag == "STRMFMT " and pending is not None:
            h = pending[0]
            n = struct.unpack_from("<H", h, 788)[0]
            record = (len(h) - CHANNEL_RECORDS) // n if n else 0
            records = [h[CHANNEL_RECORDS + k * record: CHANNEL_RECORDS + (k + 1) * record] for k in range(n)]
            unit = chr(body[2]).strip() if len(body) > 2 and body[2] else ""
            streams.append(McdStream(
                tag=h[792:800].decode("latin-1"), type_name=_cstring(h[2:12]),
                sampling_frequency=struct.unpack_from("<I", h, 784)[0] / 1000.0,
                channel_ids=[_cstring(r[790:798]) for r in records],
                hwids=[struct.unpack_from("<I", r, 8)[0] for r in records],
                unit=unit, ad_zero=struct.unpack_from("<H", body, 6)[0],
                resolution=struct.unpack_from("<d", body, 8)[0] if len(body) >= 16 else 0.0,
                samples_per_block=struct.unpack_from("<H", body, 18)[0] if len(body) >= 20 else 0))
            pending = None
    return version, streams


@dataclass
class McdIndex:
    """Positions of the data chunks of one file."""

    version: tuple[int, ...]
    streams: list[McdStream]
    group_times: np.ndarray  # uint64, ns since 1601, start of each block group
    offsets: dict[str, np.ndarray]  # stream tag -> file offset of the samples of each block
    events: np.ndarray  # trigger events: time (ns since 1601), stream number, value, number, trial, stimulus
    block_ns: int  # duration of a block
    sweeps: int = 1
    truncated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def start(self) -> datetime:
        return EPOCH + timedelta(microseconds=int(self.group_times[0]) // 1000)

    @property
    def end_ns(self) -> int:
        return int(self.group_times[-1]) + self.block_ns


EVENT_DTYPE = np.dtype([("time", "<u8"), ("stream", "<i4"), ("value", "<i4"), ("number", "<i4"),
                        ("trial", "<i8"), ("stimulus", "<i8")])


def _parse_events(body: bytes, stream_number: int) -> list[tuple]:
    """Trigger events of a trigger chunk body (after the 16-byte block times)."""
    payload = body[16:]
    if not payload:
        return []
    size = next((s for s in (22, 14, 10) if len(payload) % s == 0), None)
    if size is None:
        raise McdFormatError(f"trigger chunk of {len(payload)} bytes")
    out = []
    for k in range(0, len(payload), size):
        time, value = struct.unpack_from("<QH", payload, k)
        number = struct.unpack_from("<H", payload, k + 10)[0] if size >= 14 else 0
        trial, stimulus = struct.unpack_from("<ii", payload, k + 14) if size == 22 else (-1, -1)
        out.append((time, stream_number, value, number, trial, stimulus))
    return out


def _read(file, offset: int, size: int) -> bytes:  # type: ignore[no-untyped-def]
    file.seek(offset)
    return file.read(size)


def scan_file(path: str | Path) -> McdIndex:
    """Index of an MCD file from its ``IDXIDX`` (a few reads), else by walking all chunks."""
    with open(path, "rb") as file:
        file_size = file.seek(0, 2)
        tag, _ = struct.unpack("<8sq", _read(file, 0, 16))
        if tag != MAGIC:
            raise McdFormatError(f"{path}: not an MC_Rack file ({tag!r})")
        tag, size = struct.unpack("<8sq", _read(file, 16, 16))
        if tag != b"LISThdr ":
            raise McdFormatError(f"{path}: no header list")
        version, streams = parse_streams(_read(file, 32, size))
        data_start = 32 + size
        tag, size = struct.unpack("<8sq", _read(file, data_start, 16))
        index_pos = struct.unpack("<q", _read(file, data_start + 16, 8))[0] if tag == b"IDX1POS " else -1
        first = data_start + 24 if tag == b"IDX1POS " else data_start
        groups = None
        sweeps = 1
        if 0 < index_pos < file_size:
            tail = _read(file, index_pos, file_size - index_pos)
            for t, start, n in _chunks(tail):
                if t == "IDXIDX  ":
                    groups = np.frombuffer(tail[start:start + n], dtype=[("time", "<u8"), ("pos", "<i8")]).copy()
                elif t == "IDXSWEEP":
                    sweeps = max(1, n // 16)
        if sweeps > 1:
            raise McdFormatError(f"{path}: triggered (sweep) recording, only continuous recordings are read")
        index = (_index_from_groups(file, streams, groups, index_pos, first) if groups is not None and len(groups)
                 else None)
        if index is None:
            logger.warning("%s: no usable chunk index, walking all chunks (slow on a network share)", Path(path).name)
            index = _walk(file, streams, first, file_size)
        index.version = version
        return index


def _index_from_groups(file, streams: list[McdStream], groups: np.ndarray, index_pos: int,  # type: ignore[no-untyped-def]
                       first: int) -> McdIndex | None:
    """Chunk positions from ``IDXIDX`` (an entry every k block groups): a span between two
    entries whose size is that of its groups with empty trigger chunks is computed, other spans
    (with trigger events) are walked. None if the layout is not as expected."""
    by_tag = {s.tag: s for s in streams}
    if not len(groups) or int(groups["pos"][0]) != first:
        return None
    ends = np.r_[groups["pos"][1:], index_pos].astype(np.int64)
    layout: list[tuple[str, int]] = []  # (tag, body size) of the first group
    pos = first
    while pos < ends[0]:
        tag_bytes, size = struct.unpack("<8sq", _read(file, pos, 16))
        tag = tag_bytes.decode("latin-1")
        if layout and tag == layout[0][0]:
            break
        layout.append((tag, size))
        pos += 16 + size
    order = [t for t, _ in layout]
    data = [t for t in order if t in by_tag and by_tag[t].kind in DATA_STREAMS]
    triggers = [t for t in order if t not in data]
    if not data or any(by_tag[t].chunk_size != n for t, n in layout if t in data) or len(set(order)) != len(order):
        return None
    others = [t for t in triggers if t[:4] != "trig"]  # e.g. averages: skipped
    group_empty = (sum(16 + by_tag[t].chunk_size for t in data) + sum(16 + n for t, n in layout if t in others)
                   + 32 * (len(triggers) - len(others)))
    times = groups["time"].astype(np.uint64)  # ns since 1601 exceed int64
    first_data = first + sum(16 + n for t, n in layout[:order.index(data[0])])
    block_start, block_end = struct.unpack("<QQ", _read(file, first_data + 16, 16))
    block_ns = int(block_end - block_start)
    offsets: dict[str, list[np.ndarray]] = {t: [] for t in data}
    group_times: list[np.ndarray] = []
    events: list[tuple] = []
    notes: list[str] = []
    for i in range(len(groups)):
        start, end = int(groups["pos"][i]), int(ends[i])
        if i + 1 < len(groups):
            m = int(round(int(times[i + 1] - times[i]) / block_ns))
        else:
            m = int(round((end - start) / group_empty))
        if m * group_empty == end - start:  # no trigger events: computed
            base = start + np.arange(m, dtype=np.int64) * group_empty
            shift = 0
            for tag in order:
                if tag in data:
                    offsets[tag].append(base + shift + 32)
                    shift += 16 + by_tag[tag].chunk_size
                else:
                    shift += 16 + dict(layout)[tag] if tag in others else 32
            group_times.append(times[i] + np.arange(m, dtype=np.uint64) * np.uint64(block_ns))
            continue
        pos, k = start, 0
        span_offsets: dict[str, list[int]] = {t: [] for t in data}
        span_times: list[int] = []
        while pos < end:  # walk the span
            tag_bytes, size = struct.unpack("<8sq", _read(file, pos, 16))
            tag = tag_bytes.decode("latin-1")
            if tag in span_offsets:
                if tag == data[0]:
                    span_times.append(struct.unpack("<Q", _read(file, pos + 16, 8))[0])
                span_offsets[tag].append(pos + 32)
            elif tag in triggers and tag not in others and size > 16:
                events += _parse_events(_read(file, pos + 16, size), int(tag[4:]) if tag[4:].isdigit() else 0)
            elif tag not in triggers:
                return None
            pos += 16 + size
            k += 1
        complete = min(len(v) for v in span_offsets.values())
        if pos != end or (max(len(v) for v in span_offsets.values()) != complete and i + 1 < len(groups)):
            return None
        if max(len(v) for v in span_offsets.values()) != complete:  # the recording stopped within its last group
            notes.append("incomplete last block group left out")
        for tag in data:
            offsets[tag].append(np.array(span_offsets[tag][:complete], np.int64))
        group_times.append(np.array(span_times[:complete], np.uint64))
    result = {t: np.concatenate(v) for t, v in offsets.items()}
    for g in (0, len(result[data[0]]) // 2, len(result[data[0]]) - 1):  # check the computed positions
        for tag, off in result.items():
            if _read(file, int(off[g]) - 32, 8).decode("latin-1") != tag:
                return None
    return McdIndex((), streams, np.concatenate(group_times).astype(np.uint64), result,
                    np.array(events, dtype=EVENT_DTYPE), block_ns, notes=notes)


def _walk(file, streams: list[McdStream], start: int, file_size: int) -> McdIndex:  # type: ignore[no-untyped-def]
    by_tag = {s.tag: s for s in streams}
    offsets: dict[str, list[int]] = {s.tag: [] for s in streams if s.kind in DATA_STREAMS}
    times: list[int] = []
    events: list[tuple] = []
    pos, truncated = start, False
    first_tag = next(iter(offsets), None)
    while pos + 16 <= file_size:
        tag_bytes, size = struct.unpack("<8sq", _read(file, pos, 16))
        tag = tag_bytes.decode("latin-1")
        if tag.startswith("IDX"):
            break
        if pos + 16 + size > file_size:
            truncated = True
            break
        stream = by_tag.get(tag)
        if stream is not None and stream.kind in DATA_STREAMS:
            if tag == first_tag:
                times.append(struct.unpack("<Q", _read(file, pos + 16, 8))[0])
            offsets[tag].append(pos + 32)
        elif stream is not None and stream.kind == "trig" and size > 16:
            events += _parse_events(_read(file, pos + 16, size), int(tag[4:]) if tag[4:].isdigit() else 0)
        pos += 16 + size
    n = min(len(v) for v in offsets.values()) if offsets else 0
    group_times = np.array(times[:n], dtype=np.uint64)
    block_ns = int(np.median(np.diff(group_times).astype(np.int64))) if n > 1 else 100_000_000  # consecutive blocks
    return McdIndex((), streams, group_times, {t: np.array(v[:n], np.int64) for t, v in offsets.items()},
                    np.array(events, dtype=EVENT_DTYPE), block_ns, truncated=truncated)


@cache
def mcd_index(path: str, size: int, mtime_ns: int) -> McdIndex:  # noqa: ARG001 (size, mtime: cache key)
    return scan_file(path)


def read_index(path: str | Path) -> McdIndex:
    stat = Path(path).stat()
    return mcd_index(str(path), stat.st_size, stat.st_mtime_ns)


# --- runs: chains of files that continue each other ------------------------------------------


@dataclass
class McdRunFiles:
    paths: list[str]

    @property
    def indexes(self) -> list[McdIndex]:
        return [read_index(p) for p in self.paths]


def _same_layout(a: McdIndex, b: McdIndex) -> bool:
    return [(s.tag, s.channel_ids, s.sampling_frequency) for s in a.streams] == \
           [(s.tag, s.channel_ids, s.sampling_frequency) for s in b.streams]


def group_files(paths: list[Path]) -> list[list[Path]]:
    """Consecutive files (by name) whose blocks continue each other form one run."""
    runs: list[list[Path]] = []
    previous: McdIndex | None = None
    for path in paths:
        try:
            index = read_index(path)
        except (McdFormatError, struct.error, OSError) as error:
            logger.warning("%s: not read (%s)", path.name, error)
            previous = None
            continue
        if not len(index.group_times):
            logger.warning("%s: no data", path.name)
            previous = None
            continue
        continues = (previous is not None and _same_layout(previous, index)
                     and abs(int(index.group_times[0]) - previous.end_ns) < previous.block_ns // 2)
        if continues:
            runs[-1].append(path)
        else:
            runs.append([path])
        previous = index
    return runs


# --- recordings ----------------------------------------------------------------------------


@cache
def _memmap(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint8, mode="r")


class McdRecordingSegment(BaseRecordingSegment):
    def __init__(self, paths: list[str], tag: str, sampling_frequency: float):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        indexes = [read_index(p) for p in paths]
        stream = next(s for s in indexes[0].streams if s.tag == tag)
        self._paths, self._tag = paths, tag
        self._n_channels, self._per_block = len(stream.channel_ids), stream.samples_per_block
        self._digital, self._ad_zero = stream.is_digital, stream.ad_zero
        self._file = np.concatenate([np.full(len(ix.offsets[tag]), k, np.int32) for k, ix in enumerate(indexes)])
        self._offsets = np.concatenate([ix.offsets[tag] for ix in indexes])

    def get_num_samples(self) -> int:
        return int(len(self._offsets) * self._per_block)

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        start = 0 if start_frame is None else int(start_frame)
        end = self.get_num_samples() if end_frame is None else int(end_frame)
        first, last = start // self._per_block, (max(end, start + 1) - 1) // self._per_block
        values = self._per_block * self._n_channels
        blocks = [np.frombuffer(_memmap(self._paths[self._file[b]]), "<u2", values, int(self._offsets[b]))
                  for b in range(first, last + 1)] if end > start else []
        raw = (np.concatenate(blocks) if blocks else np.zeros(0, "<u2")).reshape(-1, self._n_channels)
        raw = raw[start - first * self._per_block: end - first * self._per_block]
        if channel_indices is not None:
            raw = raw[:, channel_indices]
        if self._digital:
            return raw.astype(np.uint16)
        return (raw.astype(np.int32) - self._ad_zero).astype(np.int16)


class McdRecordingExtractor(BaseRecording):
    """One stream of a run (its chain of files): int16 counts (sample - AD zero) with the
    resolution as gain (µV per count); the digital stream as uint16 words (gain 1)."""

    def __init__(self, paths: list[str], tag: str):
        stream = next(s for s in read_index(paths[0]).streams if s.tag == tag)
        dtype = np.dtype("uint16") if stream.is_digital else np.dtype("int16")
        BaseRecording.__init__(self, stream.sampling_frequency, list(stream.channel_ids), dtype)
        self.add_recording_segment(McdRecordingSegment(paths, tag, stream.sampling_frequency))
        scale = {"V": 1e6, "": 1.0}.get(stream.unit, 1e6)
        self.set_channel_gains(1.0 if stream.is_digital else stream.resolution * scale)
        self.set_channel_offsets(0.0)
        self.set_property("hwid", np.array(stream.hwids, dtype=np.int64))
        self._kwargs = {"paths": list(paths), "tag": tag}


@cache
def _extractor(paths: tuple[str, ...], tag: str) -> McdRecordingExtractor:
    return McdRecordingExtractor(list(paths), tag)


# --- digital transitions (cached) ------------------------------------------------------------


def _transitions_cache_path(path: str, tag: str, mask: int) -> Path:
    from ephysalis.legacy.daq import cache_dir

    stat = Path(path).stat()
    key = f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{tag}|{mask}|{_CACHE_VERSION}"
    return cache_dir() / "mcd-digital" / f"{Path(path).stem}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def digital_transitions(path: str, tag: str, mask: int = 0xFFFF) -> tuple[np.ndarray, np.ndarray, int]:
    """(sample of each change of the digital word's bits ``mask`` within the file, new masked
    word, first masked word)."""
    cached = _transitions_cache_path(path, tag, mask)
    if cached.exists():
        with np.load(cached) as data:
            return data["samples"], data["words"], int(data["first"])
    recording = McdRecordingExtractor([path], tag)
    n, chunk = recording.get_num_samples(), 2_500_000
    samples, words, first, previous = [], [], None, None
    for begin in range(0, n, chunk):
        word = recording.get_traces(start_frame=begin, end_frame=min(n, begin + chunk))[:, 0].astype(np.int64) & mask
        if first is None:
            first = previous = int(word[0])
        change = np.flatnonzero(np.diff(np.r_[previous, word]) != 0)
        samples.append(change + begin)
        words.append(word[change])
        previous = int(word[-1])
    out = (np.concatenate(samples).astype(np.int64) if samples else np.zeros(0, np.int64),
           np.concatenate(words).astype(np.uint16) if words else np.zeros(0, np.uint16), int(first or 0))
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(temporary, samples=out[0], words=out[1], first=np.array(out[2]))
    temporary.replace(cached)
    return out


# --- format ---------------------------------------------------------------------------------


def _mcd_files(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mcd"),
                  key=lambda p: (re.sub(r"\d{4}$", "", p.stem), p.stem))


class McdFormat(InputFormat):
    name: ClassVar[str] = "mcd"

    def can_read(self, session_path: Path) -> bool:
        return any(p.suffix.lower() == ".mcd" for p in session_path.iterdir() if p.is_file())

    def open_session(self, session_path: Path) -> Session:
        runs: list[Run] = []
        chains = sorted(group_files(_mcd_files(session_path)), key=lambda files: read_index(files[0]).start)
        for files in chains:
            indexes = [read_index(p) for p in files]
            streams = indexes[0].streams
            data = [s for s in streams if s.kind in DATA_STREAMS]
            aux = next((s.tag for s in data if s.is_digital), None) or next((s.tag for s in data if s.kind == "anlg"), None)
            main = next(s for s in data if s.tag == (aux or data[0].tag)) if data else None
            n_blocks = sum(len(ix.group_times) for ix in indexes)
            runs.append(Run(
                name=files[0].stem, path=session_path, index=len(runs), format=self, start_time=indexes[0].start,
                duration_s=n_blocks * (main.samples_per_block / main.sampling_frequency) if main else None,
                aux_stream=aux, streams=[s.tag for s in data],
                extra={"files": [str(p) for p in files], "mcs_version": list(indexes[0].version),
                       "stream_types": {s.tag: s.type_name for s in streams},
                       "truncated": [str(p) for p, ix in zip(files, indexes) if ix.truncated]},
            ))
        return Session(name=session_path.name, path=session_path, format=self, runs=runs)

    def default_aux_channels(self, run: Run | None = None) -> AuxChannelMap:
        from ephysalis.session_config import AuxChannelMap

        return AuxChannelMap(analog={}, digital={})

    def get_recording(self, run: Run, kind: StreamKind, probe_id: str | None = None) -> BaseRecording:
        if kind == StreamKind.AUX:
            tag = run.aux_stream
        else:
            if probe_id is None:
                raise ValueError(f"probe_id is required for {kind} streams")
            try:
                tag = run.probes[probe_id].streams[kind]
            except KeyError:
                raise KeyError(f"Run {run.name} has no {kind} stream for {probe_id}") from None
        if tag is None:
            raise KeyError(f"Run {run.name} has no auxiliary stream")
        return _extractor(tuple(run.extra["files"]), tag)

    def _streams(self, run: Run) -> dict[str, McdStream]:
        return {s.tag: s for s in read_index(run.extra["files"][0]).streams}

    def _analog_stream(self, run: Run) -> str | None:
        return next((t for t, s in self._streams(run).items() if s.kind == "anlg"), None)

    def aux_analog_recording(self, run: Run) -> BaseRecording:
        """The analog inputs (``anlg`` stream), else the digital word."""
        tag = self._analog_stream(run)
        return _extractor(tuple(run.extra["files"]), tag) if tag else self.get_recording(run, StreamKind.AUX)

    def read_aux_analog(self, run: Run, index: int) -> np.ndarray:
        tag = self._analog_stream(run)
        if tag is None:
            raise KeyError(f"Run {run.name} has no analog inputs")
        return _extractor(tuple(run.extra["files"]), tag).get_traces(channel_ids=None)[:, index]

    def aux_analog_gain_v(self, run: Run, index: int) -> float:
        tag = self._analog_stream(run)
        if tag is None:
            raise KeyError(f"Run {run.name} has no analog inputs")
        return float(_extractor(tuple(run.extra["files"]), tag).get_channel_gains()[index]) * 1e-6

    def clock_maps(self, run: Run, reference: str | None = None) -> RunSync | None:
        """All streams of a USB-ME device share its clock; the block times place the files of a run."""
        from ephysalis.sync import AUX, ClockMap, RunSync

        indexes = [read_index(p) for p in run.extra["files"]]
        t0 = indexes[0].group_times[0]
        times = (np.concatenate([ix.group_times for ix in indexes]) - t0).astype(np.int64)
        streams = {pid: info.streams[StreamKind.AP] for pid, info in run.probes.items() if StreamKind.AP in info.streams}
        if run.aux_stream:
            streams[AUX] = run.aux_stream
        if not streams:  # neither a digital stream nor a probe: the run's first stream
            streams[AUX] = run.streams[0]
        by_tag = self._streams(run)
        maps = {}
        for key, tag in streams.items():
            stream = by_tag[tag]
            samples = np.arange(len(times)) * stream.samples_per_block
            maps[key] = ClockMap(samples.astype(float), times / 1e9, stream.sampling_frequency,
                                 rate=stream.sampling_frequency)
        return RunSync(reference=reference or next(iter(streams)), maps=maps)

    def _aux_sample(self, run: Run, times_ns: np.ndarray) -> np.ndarray:
        t0 = read_index(run.extra["files"][0]).group_times[0]
        fs = self.aux_sampling_frequency(run)
        elapsed = (np.asarray(times_ns, dtype=np.uint64) - t0).astype(np.int64)  # ns since 1601 exceed int64
        return np.floor(elapsed * fs / 1e9 + 1e-6).astype(np.int64)

    def read_ttl_events(self, run: Run, lines: Collection[int] | None = None) -> pd.DataFrame:
        """Changes of the bits of the digital word (line = bit + 1; ``lines``: only these, which
        also reads only their bits: unconnected inputs of the USB-ME toggle all the time)."""
        if run.aux_stream is None or not self._streams(run)[run.aux_stream].is_digital:
            return pd.DataFrame({"aux_sample": np.zeros(0, np.int64), "line": np.zeros(0, np.int64),
                                 "state": np.zeros(0, np.int64)})
        bits = range(16) if lines is None else sorted(line - 1 for line in lines if 1 <= line <= 16)
        mask = sum(1 << bit for bit in bits)
        parts, offset, previous = [], 0, None
        for path in run.extra["files"]:
            samples, words, first = digital_transitions(path, run.aux_stream, mask) if mask else (
                np.zeros(0, np.int64), np.zeros(0, np.uint16), 0)
            if previous is not None and first != previous:  # a change at the file boundary
                samples, words = np.r_[0, samples], np.r_[first, words].astype(np.uint16)
            before = np.r_[previous if previous is not None else first, words[:-1]].astype(np.int64)
            changed = before ^ words.astype(np.int64)
            for bit in bits:
                hit = np.flatnonzero((changed >> bit) & 1)
                parts.append(pd.DataFrame({"aux_sample": samples[hit] + offset, "line": bit + 1,
                                           "state": (words[hit].astype(np.int64) >> bit) & 1}))
            if len(words):
                previous = int(words[-1])
            elif previous is None:
                previous = first
            offset += McdRecordingExtractor([path], run.aux_stream).get_num_samples()
        events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["aux_sample", "line", "state"])
        return events.astype(np.int64).sort_values(["aux_sample", "line"], kind="stable", ignore_index=True)

    def read_aux_digital(self, run: Run, bit: int) -> np.ndarray:
        word = self.get_recording(run, StreamKind.AUX).get_traces()[:, 0]
        return ((word.astype(np.int64) >> bit) & 1).astype(bool)

    def read_messages(self, run: Run) -> pd.DataFrame:
        return pd.DataFrame({"aux_sample": np.zeros(0, np.int64), "text": np.zeros(0, str)})

    def read_trigger_events(self, run: Run) -> pd.DataFrame:
        """All trigger events: ``aux_sample``, ``stream`` (number of the trigger stream),
        ``value``, ``number``, ``trial``, ``stimulus`` (-1 without trial synchronisation)."""
        events = np.concatenate([read_index(p).events for p in run.extra["files"]])
        return pd.DataFrame({"aux_sample": self._aux_sample(run, events["time"]), "stream": events["stream"],
                             "value": events["value"], "number": events["number"], "trial": events["trial"],
                             "stimulus": events["stimulus"]})

    def read_trial_descriptors(self, run: Run) -> pd.DataFrame:
        """Trigger events with trial synchronisation (trial and stimulus number from the
        stimulus PC, as the ``TD01`` records of DAQ files): ``aux_sample``, ``trial_number``,
        ``stimulus_number``."""
        events = self.read_trigger_events(run)
        events = events[events.trial >= 0]
        return pd.DataFrame({"aux_sample": events.aux_sample.to_numpy(np.int64),
                             "trial_number": events.trial.to_numpy(np.int64),
                             "stimulus_number": events.stimulus.to_numpy(np.int64)})
