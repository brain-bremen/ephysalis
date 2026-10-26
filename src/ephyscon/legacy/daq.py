"""Reader for the DAQ files (``*.daq``) of the acquisition program of the Brain Research Institute,
University of Bremen (ADC64 and MC-128 cards, 1999-2006).

A session folder contains one ``.daq`` file per recording block. Each contiguous stretch of a
file (a file has several when the recording was paused) becomes one
:class:`~ephyscon.session.Run`, named after the file (``<stem>_part<k>`` for files with
several). Runs are ordered by the start time in the file header.

Format (reverse-documented from Michael Borisov's DAQ -> HDF5 converter, V2.7): little-endian
packed records ``char tag[4], int32 size`` (size including this header), times are int64 ns of
the DAQ clock (which need not restart at 0 with a new file).

=================  ==========================================================================
file version       records
=================  ==========================================================================
1.0, 1.1, 2.0      header ``VER1 DATE NAME BORD FCR1`` (one board, all channels at one rate,
                   which is not stored: it is the spacing of the ``CR01`` times); data
                   ``CR01`` (time + one sample of every channel), ``EV01`` (time, ±line,
                   line states), ``TD01`` (time, trial and stimulus number)
3.1, 3.2, 3.3      header ``VER1 DATE NAME [RECP] BORD EDR1 CDR1`` (nTrodes = groups of
                   channels with one rate; channel descriptors with the sample rate); data
                   ``CR02`` (time of the first sample, nTrode id, samples ``[sample][channel]``),
                   ``EV02`` (time, ±line), ``TD01``
=================  ==========================================================================

Version 3.0 files are known to be corrupt and are not read. Also handled: an ``FCR1`` size
field truncated to its low byte (version 1.0, 2000), a zero-filled end of a preallocated file
(version 1.x/2.0), a last record cut off by the end of the file, foreign bytes inside a
version 3 file (skipped up to the next chain of valid records, listed in ``run.extra["damaged"]``)
and ``CR02`` records whose time lies before the end of the previous record (version 3.1;
taken as continuation, as the original converter did). Spike nTrodes (waveform snippets) are
not read.

Streams: the continuous channels of each board (e.g. ``ADC64 F/H - Board 1`` and
``MC-128 - Board 1``; a board with several sample rates gets one stream per rate). The board
of a channel follows from the channel counts in the board names when they add up to the
number of channels (ADC64 = channels 0-63, MC-128 = 64-191), else from the restarts of the
board channel numbers. Channel ids are ``AI<n>`` with ``n`` the DAQ channel (0-based). Samples
are raw ADC counts in int16, 12-bit cards left-aligned (multiples of 16), so the gain is the
ADC range / 65536 per count (divided by the stored amplification where it is set). This is the
voltage at the ADC input; amplifier gains are not stored.

Probes are not stored in the files: define them in ``session.toml`` (``stream`` and
``channels``, see :mod:`ephyscon.io.probes`). The first board's stream is the aux stream: its
channels that belong to no probe are the analog inputs (``read_aux_analog``), the trigger lines
of ``EV01``/``EV02`` its TTL lines, ``TD01`` its trial descriptors (:meth:`DaqFormat.read_trial_descriptors`).

Clocks: the boards share no clock. In the 2006 setup the MC-128 (board 2) runs about 0.38 %
fast against the ADC64 and records copies of trigger lines 1-16 as lines 17-32 on its own clock;
the matched copies give board 2 time = offset + rate x board 1 time (:func:`board_fit`, as the
lab's ``synchro_timestamp_adj.m``), from which the clock maps (:meth:`DaqFormat.clock_maps`) and
the times of the copied lines follow. Without copies the boards' times are taken as one clock.

Indexing a file reads the head of every record, which takes minutes for GB files on a network
share; the index is therefore cached on disk (:func:`cache_dir`, ``EPHYSCON_CACHE_DIR``), keyed by
the file's path, size and modification time.
"""

from __future__ import annotations

from collections.abc import Collection

import hashlib
import logging
import os
import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from ephyscon.io.base import InputFormat, keep_lines
from ephyscon.session import Run, Session, StreamKind

if TYPE_CHECKING:
    from ephyscon.session_config import AuxChannelMap
    from ephyscon.sync import RunSync

logger = logging.getLogger(__name__)

_RECORD = struct.Struct("<4si")
_CR01 = int.from_bytes(b"CR01", "little")
_DATA_TAGS = re.compile(rb"CR02|EV02|TD01")
# plausible time of the first record after damaged bytes, relative to the last good record
_RESYNC_WINDOW_NS = (-10_000_000_000, 600_000_000_000)
_INDEX_CACHE_VERSION = 1
COPY_OFFSET = 16  # 2006 setup: trigger line L of board 1 is copied to line L + 16 on board 2
FULL_SCALE_COUNTS = 65536  # samples use the int16 range, also for 12-bit cards (left-aligned)


class DaqFormatError(ValueError):
    """A DAQ file that cannot be read."""


@dataclass(frozen=True)
class DaqChannel:
    global_chan: int
    board_chan: int
    adc_bits: int
    max_voltage: float
    min_voltage: float
    amplification: float  # 0: not stored
    sample_rate: float  # Hz, 0: not stored (file versions 1.x, 2.0)

    @property
    def volts_per_count(self) -> float:
        gain = self.amplification if self.amplification > 0 else 1.0
        return (self.max_voltage - self.min_voltage) / FULL_SCALE_COUNTS / gain


@dataclass(frozen=True)
class DaqNtrode:
    id: int
    spike: bool  # spike waveform snippets instead of continuous samples
    first_chan: int
    n_chan: int


@dataclass
class DaqHeader:
    program_version: tuple[int, int]
    file_version: tuple[int, int]
    date: tuple[int, ...]  # (year, month, day, hour, minute, second), local time of the acquisition PC
    user: str
    subject: str
    boards: list[str] = field(default_factory=list)
    channels: list[DaqChannel] = field(default_factory=list)
    ntrodes: list[DaqNtrode] = field(default_factory=list)  # version 3; version 2: one nTrode of all channels
    recording_type: int | None = None  # RECP (3.3): 0 continuous, 1 trial based
    start_trigger: int | None = None
    stop_trigger: int | None = None
    data_offset: int = 0

    @property
    def start(self) -> datetime | None:
        try:
            year, month, day, hour, minute, second = self.date
            return datetime(year, month, day, hour, minute, second)
        except (TypeError, ValueError):
            return None


def _cstr(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1")


def _expect(buf, offset: int, tag: bytes) -> tuple[int, int]:
    """Size and body offset of the record ``tag`` at ``offset``."""
    if offset + 8 > len(buf):
        raise DaqFormatError(f"file ends before the {tag.decode()} record")
    found, size = _RECORD.unpack_from(buf, offset)
    if found != tag:
        raise DaqFormatError(f"offset {offset}: expected {tag!r}, found {found!r}")
    return size, offset + 8


def read_header(buf) -> DaqHeader:
    """Header records of a DAQ file (``buf``: bytes-like, e.g. a memmap of the file)."""
    size, body = _expect(buf, 0, b"VER1")
    pmaj, pmin, fmaj, fmin = struct.unpack_from("<4h", buf, body)
    offset = size
    v2 = (fmaj, fmin) in ((1, 0), (1, 1), (2, 0))
    if not v2 and not (fmaj == 3 and fmin in (1, 2, 3)):
        raise DaqFormatError(f"DAQ file version {fmaj}.{fmin} is not supported")
    size, body = _expect(buf, offset, b"DATE")
    date = struct.unpack_from("<6h", buf, body)
    offset += size
    size, body = _expect(buf, offset, b"NAME")
    header = DaqHeader((pmaj, pmin), (fmaj, fmin), date, _cstr(bytes(buf[body:body + 40])),
                       _cstr(bytes(buf[body + 40:body + 80])))
    offset += size
    if fmaj == 3 and fmin == 3:
        size, body = _expect(buf, offset, b"RECP")
        header.recording_type, header.start_trigger, header.stop_trigger, _ = struct.unpack_from("<Hbbi", buf, body)
        offset += size
    size, body = _expect(buf, offset, b"BORD")
    (n,) = struct.unpack_from("<i", buf, body)
    header.boards = [_cstr(bytes(buf[body + 4 + 20 * i:body + 24 + 20 * i])) for i in range(n)]
    offset += size
    if v2:
        size, body = _expect(buf, offset, b"FCR1")
        vmax, vmin, n, bits = struct.unpack_from("<ffhh", buf, body)
        if size != 20 + 6 * n:
            if size != (20 + 6 * n) & 0xFF:  # 2000 files: size field truncated to its low byte
                raise DaqFormatError(f"FCR1 record of {size} bytes for {n} channels")
            size = 20 + 6 * n
        for i in range(n):
            number, amplification = struct.unpack_from("<hi", buf, body + 12 + 6 * i)
            header.channels.append(DaqChannel(number, number, bits, vmax, vmin, float(amplification), 0.0))
        header.ntrodes = [DaqNtrode(0, False, 0, n)]
        header.data_offset = offset + size
        return header
    size, body = _expect(buf, offset, b"EDR1")
    (n,) = struct.unpack_from("<i", buf, body)
    for i in range(n):
        number, kind, first, n_chan = struct.unpack_from("<hbhb", buf, body + 4 + 12 * i)
        header.ntrodes.append(DaqNtrode(number, kind == 1, first, n_chan))
    offset += size
    size, body = _expect(buf, offset, b"CDR1")
    (n,) = struct.unpack_from("<i", buf, body)
    header.channels = [DaqChannel(*struct.unpack_from("<hhh4f", buf, body + 4 + 22 * i)) for i in range(n)]
    header.data_offset = offset + size
    return header


# --- record index ---------------------------------------------------------------------------


@dataclass
class _Pieces:
    """Where the samples of one continuous nTrode are: runs of samples with a fixed byte stride."""

    channels: tuple[int, ...]  # DAQ channels (columns)
    data_offsets: np.ndarray  # byte offset of the first sample of each piece
    n_samples: np.ndarray
    strides: np.ndarray  # bytes from one sample to the next
    times: np.ndarray  # ns of the first sample of each piece
    period_ns: int
    starts: np.ndarray = field(init=False)  # index of the first sample of each piece

    def __post_init__(self) -> None:
        self.starts = np.r_[0, np.cumsum(self.n_samples)[:-1]].astype(np.int64)

    @property
    def total(self) -> int:
        return int(self.n_samples.sum())

    def regions(self) -> list[tuple[int, int, int]]:
        """Contiguous stretches ``(first sample, n samples, time of the first sample)``: a piece
        starting later than the end of the previous one starts a new stretch (pause)."""
        regions: list[tuple[int, int, int]] = []
        expected = None
        for start, n, t in zip(self.starts.tolist(), self.n_samples.tolist(), self.times.tolist()):
            if expected is None or t > expected:
                regions.append((start, n, t))
            else:
                first, count, t0 = regions[-1]
                regions[-1] = (first, count + n, t0)
            expected = (t if expected is None or t > expected else expected) + n * self.period_ns
        return regions

    def sample_at(self, times_ns: np.ndarray | float) -> np.ndarray:
        """Fractional sample indices at DAQ times (nominal clock of the nTrode): counted from
        the start of the last piece that starts at or before each time."""
        times_ns = np.asarray(times_ns, dtype=float)
        k = np.maximum(np.searchsorted(self.times, times_ns, side="right") - 1, 0)
        return self.starts[k] + (times_ns - self.times[k]) / self.period_ns

    def time_at(self, samples: np.ndarray) -> np.ndarray:
        """DAQ times (ns) of sample indices: start time of their piece plus the sample period."""
        samples = np.asarray(samples, dtype=np.int64)
        k = np.maximum(np.searchsorted(self.starts, samples, side="right") - 1, 0)
        return self.times[k] + (samples - self.starts[k]) * float(self.period_ns)

    def read(self, buf, start: int, stop: int, columns: list[int]) -> np.ndarray:
        out = np.empty((max(stop - start, 0), len(columns)), dtype=np.int16)
        if stop <= start:
            return out
        n_chan = len(self.channels)
        first = int(np.searchsorted(self.starts, start, side="right")) - 1
        last = int(np.searchsorted(self.starts, stop, side="left"))
        for k in range(first, last):
            a, n = int(self.starts[k]), int(self.n_samples[k])
            lo, hi = max(start, a), min(stop, a + n)
            if hi <= lo:
                continue
            piece = np.ndarray((n, n_chan), dtype="<i2", buffer=buf, offset=int(self.data_offsets[k]),
                               strides=(int(self.strides[k]), 2))
            out[lo - start:hi - start] = piece[lo - a:hi - a][:, columns]
        return out


@dataclass
class DaqIndex:
    path: Path
    header: DaqHeader
    pieces: dict[int, _Pieces]  # continuous nTrode id -> pieces
    events: np.ndarray  # (n, 2) int64: time ns, ±line (1-based)
    trials: np.ndarray  # (n, 3) int64: time ns, trial number, stimulus number
    damaged: list[tuple[int, int]] = field(default_factory=list)  # (byte offset, bytes skipped)
    truncated_bytes: int = 0  # missing at the end of a cut-off last record
    zero_tail: int | None = None  # offset from which the file is zero bytes


class _FileBytes:
    """Bytes of a file by small reads, for scanning the record heads of version 3 files: on a
    network share a memmap reads the whole page around every record head, i.e. most of the file."""

    def __init__(self, path: str):
        self._file = open(path, "rb")  # noqa: SIM115 (closed by close())
        self._size = Path(path).stat().st_size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key: slice) -> bytes:
        self._file.seek(key.start)
        return self._file.read(key.stop - key.start)

    def close(self) -> None:
        self._file.close()


def _next_valid(buf, offset: int, n_chan: dict[int, int], t_last: int, chain: int = 3) -> int | None:
    """First offset > ``offset`` where ``chain`` valid records follow each other, the first
    with a time plausible relative to ``t_last``."""

    def record(at: int) -> tuple[int, int] | None:
        if at + 20 > len(buf):
            return None
        head = bytes(buf[at:at + 20])
        tag, size = _RECORD.unpack_from(head)
        if tag == b"CR02":
            t, nid = struct.unpack_from("<qH", head, 8)
            ok = nid in n_chan and size > 18 and (size - 18) % (2 * n_chan[nid]) == 0
        elif tag == b"EV02":
            ok, t = size == 20, struct.unpack_from("<q", head, 8)[0]
        elif tag == b"TD01":
            ok, t = size == 32 and at + 32 <= len(buf), struct.unpack_from("<q", head, 8)[0]
        else:
            return None
        return (size, t) if ok else None

    block = 1 << 20
    position = offset + 1
    while position < len(buf):
        for match in _DATA_TAGS.finditer(bytes(buf[position:position + block + 4])):
            candidate = position + match.start()
            first = record(candidate)
            if first is None or not _RESYNC_WINDOW_NS[0] <= first[1] - t_last <= _RESYNC_WINDOW_NS[1]:
                continue
            at = candidate
            for _ in range(chain):
                r = record(at)
                if r is None:
                    break
                at += r[0]
            else:
                return candidate
        position += block
    return None


def _scan_v3(buf, header: DaqHeader) -> DaqIndex:
    n_chan = {n.id: n.n_chan for n in header.ntrodes}
    continuous = {n.id: n for n in header.ntrodes if not n.spike}
    found: dict[int, tuple[list[int], list[int], list[int]]] = {i: ([], [], []) for i in continuous}
    events, trials, damaged = [], [], []
    truncated = 0
    offset, end, t_last = header.data_offset, len(buf), 0
    while offset + 8 <= end:
        head = bytes(buf[offset:offset + 32])
        tag, size = _RECORD.unpack_from(head)
        if tag == b"CR02" and offset + 18 <= end:
            t, nid = struct.unpack_from("<qH", head, 8)
            if nid in n_chan and size > 18 and (size - 18) % (2 * n_chan[nid]) == 0:
                available = min(size, end - offset) - 18
                truncated += size - 18 - available
                if nid in found and available >= 2 * n_chan[nid]:
                    offsets, ns, times = found[nid]
                    offsets.append(offset + 18)
                    ns.append(available // (2 * n_chan[nid]))
                    times.append(t)
                t_last = t
                offset += size
                continue
        elif tag == b"EV02" and size == 20 and offset + 20 <= end:
            t, line = struct.unpack_from("<qi", head, 8)
            events.append((t, line))
            t_last, offset = t, offset + size
            continue
        elif tag == b"TD01" and size == 32 and offset + 32 <= end:
            t, trial, stim = struct.unpack_from("<qii", head, 8)
            trials.append((t, trial, stim))
            t_last, offset = t, offset + size
            continue
        if end - offset < 34 and tag in (b"CR02", b"EV02", b"TD01"):  # last record cut off
            truncated += end - offset
            break
        resume = _next_valid(buf, offset, n_chan, t_last)
        damaged.append((offset, (resume if resume is not None else end) - offset))
        if resume is None:
            break
        offset = resume
    pieces = {}
    for nid, (offsets, ns, times) in found.items():
        if not offsets:
            continue
        nt = continuous[nid]
        rate = header.channels[nt.first_chan].sample_rate
        pieces[nid] = _Pieces(tuple(range(nt.first_chan, nt.first_chan + nt.n_chan)), np.array(offsets, np.int64),
                              np.array(ns, np.int64), np.full(len(offsets), 2 * nt.n_chan, np.int64),
                              np.array(times, np.int64), int(round(1e9 / rate)))
    return DaqIndex(Path(), header, pieces, np.array(events, np.int64).reshape(-1, 2),
                    np.array(trials, np.int64).reshape(-1, 3), damaged, truncated)


def _scan_v2(buf, header: DaqHeader) -> DaqIndex:
    """Version 1.x / 2.0: runs of consecutive ``CR01`` records are found with strided views
    (one record per sample frame), the other records one by one."""
    n_chan = len(header.channels)
    record_bytes = 16 + 2 * n_chan
    end = len(buf)
    runs: list[tuple[int, np.ndarray]] = []  # (offset of the first CR01 record, frame times)
    events, trials = [], []
    zero_tail, truncated = None, 0
    offset = header.data_offset
    while offset + 8 <= end:
        tag, size = _RECORD.unpack_from(buf, offset)
        if tag == b"CR01" and size == record_bytes:
            n = 0
            window = 4096
            while True:
                m = min(window, (end - offset) // record_bytes - n)
                if m <= 0:
                    break
                view = np.ndarray((m, 2), dtype="<u4", buffer=buf, offset=offset + n * record_bytes, strides=(record_bytes, 4))
                ok = (view[:, 0] == _CR01) & (view[:, 1] == record_bytes)
                k = int(np.argmin(ok)) if not ok.all() else m
                n += k
                if k < m:
                    break
                window *= 2
            times = np.ndarray((n,), dtype="<i8", buffer=buf, offset=offset + 8, strides=(record_bytes,)).copy()
            runs.append((offset, times))
            offset += n * record_bytes
            continue
        if tag == b"EV01" and size == 24 and offset + 24 <= end:
            events.append(struct.unpack_from("<qi", buf, offset + 8))
        elif tag == b"TD01" and size == 32 and offset + 32 <= end:
            trials.append(struct.unpack_from("<qii", buf, offset + 8))
        elif tag == b"\0\0\0\0" and not np.any(np.asarray(buf[offset:], dtype=np.uint8)):
            zero_tail = offset
            break
        elif tag in (b"CR01", b"EV01", b"TD01") and offset + size > end:
            truncated = end - offset
            break
        else:
            raise DaqFormatError(f"offset {offset}: unexpected record {tag!r}")
        offset += size
    if not runs:
        raise DaqFormatError("no samples (CR01 records)")
    times = np.concatenate([t for _, t in runs])
    steps = np.diff(times)
    period = int(np.median(steps)) if steps.size else 1_000_000
    if np.any(steps < period):
        raise DaqFormatError(f"CR01 sample interval not constant ({int(steps.min())} ns < {period} ns)")
    # one piece per stretch of records with consecutive times inside each run of CR01 records
    offsets, ns, starts_t = [], [], []
    for first, t in runs:
        breaks = np.r_[0, np.flatnonzero(np.diff(t) != period) + 1, t.size]
        for a, b in zip(breaks[:-1], breaks[1:]):
            offsets.append(first + int(a) * record_bytes + 16)
            ns.append(int(b - a))
            starts_t.append(int(t[a]))
    pieces = _Pieces(tuple(range(n_chan)), np.array(offsets, np.int64), np.array(ns, np.int64),
                     np.full(len(ns), record_bytes, np.int64), np.array(starts_t, np.int64), period)
    return DaqIndex(Path(), header, {0: pieces}, np.array(events, np.int64).reshape(-1, 2),
                    np.array(trials, np.int64).reshape(-1, 3), [], truncated, zero_tail)


def _version(path: str) -> tuple[int, int]:
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


@cache
def _memmap_of(path: str, version: tuple[int, int]) -> np.memmap:
    return np.memmap(path, dtype=np.uint8, mode="r")


def _memmap(path: str) -> np.memmap:
    return _memmap_of(path, _version(path))


def read_index(path: str) -> DaqIndex:
    """Header and record index of a DAQ file (cached per process while the file is unchanged)."""
    return _read_index(path, _version(path))


def cache_dir() -> Path:
    """Folder of ephyscon's caches of input files: ``EPHYSCON_CACHE_DIR``, else
    ``%LOCALAPPDATA%/ephyscon`` (Windows) or ``$XDG_CACHE_HOME/ephyscon`` (``~/.cache``)."""
    if os.environ.get("EPHYSCON_CACHE_DIR"):
        return Path(os.environ["EPHYSCON_CACHE_DIR"])
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "ephyscon"


def _index_cache_path(path: str, version: tuple[int, int]) -> Path:
    key = f"{Path(path).resolve()}|{version[0]}|{version[1]}|{_INDEX_CACHE_VERSION}"
    return cache_dir() / "daq-index" / f"{Path(path).stem}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def _save_index(index: DaqIndex, target: Path) -> None:
    arrays: dict[str, np.ndarray] = {
        "events": index.events, "trials": index.trials, "damaged": np.array(index.damaged, np.int64).reshape(-1, 2),
        "scalars": np.array([index.truncated_bytes, -1 if index.zero_tail is None else index.zero_tail], np.int64),
        "ntrodes": np.array(sorted(index.pieces), np.int64)}
    for nid, pieces in index.pieces.items():
        arrays[f"p{nid}"] = np.stack([pieces.data_offsets, pieces.n_samples, pieces.strides, pieces.times])
        arrays[f"c{nid}"] = np.array([pieces.period_ns, *pieces.channels], np.int64)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)  # type: ignore[arg-type]  # numpy's stub types the keyword arrays as bool
    temporary.replace(target)


def _load_index(source: Path, header: DaqHeader) -> DaqIndex:
    with np.load(source, allow_pickle=False) as data:
        pieces = {}
        for nid in data["ntrodes"].tolist():
            offsets, n, strides, times = data[f"p{nid}"]
            period, *channels = data[f"c{nid}"].tolist()
            pieces[nid] = _Pieces(tuple(channels), offsets, n, strides, times, period)
        truncated, zero_tail = data["scalars"].tolist()
        return DaqIndex(Path(), header, pieces, data["events"], data["trials"],
                        [tuple(d) for d in data["damaged"].tolist()], truncated, None if zero_tail < 0 else zero_tail)


@cache
def _read_index(path: str, version: tuple[int, int]) -> DaqIndex:
    """Scanning a file means reading every record head (minutes for GB files on a network
    share), so the index is also kept on disk (:func:`cache_dir`)."""
    with open(path, "rb") as file:
        header = read_header(file.read(1 << 20))  # the header records are a few kB
    cached = _index_cache_path(path, version)
    if cached.exists():
        try:
            index = _load_index(cached, header)
            index.path = Path(path)
            return index
        except (OSError, KeyError, ValueError) as error:
            logger.warning("%s: index cache %s not read (%s)", Path(path).name, cached, error)
    logger.info("%s: indexing the records (%.1f GB; cached for later runs)", Path(path).name, version[0] / 1e9)
    if header.file_version[0] < 3:
        index = _scan_v2(_memmap(path), header)  # one record per sample frame: the whole file is read anyway
    else:
        heads = _FileBytes(path)
        try:
            index = _scan_v3(heads, header)
        finally:
            heads.close()
    index.path = Path(path)
    try:
        _save_index(index, cached)
    except OSError as error:
        logger.warning("%s: index not cached (%s)", Path(path).name, error)
    for offset, n in index.damaged:
        logger.warning("%s: %d unreadable bytes at offset %d skipped", index.path.name, n, offset)
    if index.truncated_bytes:
        logger.warning("%s: the file ends inside a record (%d bytes missing)", index.path.name, index.truncated_bytes)
    return index


def board_fit(events: np.ndarray, tolerance_ns: float = 20e6, rate0: float = 1.0038) -> dict | None:
    """Board 2 time = ``offset_ns + rate`` x board 1 time, from the copies of trigger lines 1-16
    on lines 17-32 (``events``: time ns, ±line), or None without copies. Port of
    ``clocksync.card_fit`` of the Fritz BIDS conversion: start from the most frequent difference
    between the first 50 line-1 edges and their copies (the first edges need not correspond),
    then match every edge to its nearest copy and refit (6 times)."""
    events = np.asarray(events, dtype=float)
    lines = np.abs(events[:, 1]) if len(events) else np.zeros(0)
    own = events[(lines >= 1) & (lines <= COPY_OFFSET)]
    copied = events[(lines > COPY_OFFSET) & (lines <= 2 * COPY_OFFSET)]
    original_line = (np.sign(copied[:, 1]) * (np.abs(copied[:, 1]) - COPY_OFFSET)).astype(int)
    copies = {int(line): np.sort(copied[original_line == line, 0]) for line in np.unique(original_line)}
    if 1 not in copies or not np.any(own[:, 1] == 1):
        return None
    a1, c1 = own[own[:, 1] == 1, 0][:50], copies[1][:50]
    d = (c1[None, :] - rate0 * a1[:, None]).ravel()
    n_near = [(np.abs(d - x) < tolerance_ns).sum() for x in d]
    offset, rate = d[int(np.argmax(n_near))], rate0
    pa, pb = np.zeros(0), np.zeros(0)
    for iteration in range(6):
        matched_a, matched_b = [], []
        for t, e in own:
            candidates = copies.get(int(e))
            if candidates is None:
                continue
            predicted = offset + rate * t
            j = int(np.searchsorted(candidates, predicted))
            for jj in (j - 1, j):
                if 0 <= jj < len(candidates) and abs(candidates[jj] - predicted) < (tolerance_ns if iteration else 300e6):
                    matched_a.append(t)
                    matched_b.append(candidates[jj])
                    break
        pa, pb = np.array(matched_a), np.array(matched_b)
        if len(pa) < 2:
            return None
        rate, offset = np.polyfit(pa, pb, 1)
    residual = (pb - (offset + rate * pa)) / 1e6
    return {"offset_ns": float(offset), "rate": float(rate), "residual_sd_ms": float(residual.std()),
            "residual_max_ms": float(np.abs(residual).max()), "n_matched": int(len(pa)), "n_edges": int(len(own))}


# --- streams --------------------------------------------------------------------------------


def channel_boards(header: DaqHeader) -> list[int]:
    """Board index of every channel (see the module docstring)."""
    n = len(header.channels)
    counts = [int(m[-1]) if (m := re.findall(r"\d+", name.split(" - ")[0])) else 0 for name in header.boards]
    if len(header.boards) == 1 or not header.channels:
        return [0] * n
    if sum(counts) == n and all(counts):
        return [b for b, count in enumerate(counts) for _ in range(count)]
    boards, board = [], 0
    for i, channel in enumerate(header.channels):
        if i and channel.board_chan <= header.channels[i - 1].board_chan:
            board += 1
        boards.append(min(board, len(header.boards) - 1))
    return boards


@dataclass(frozen=True)
class DaqStream:
    name: str
    board: int
    ntrodes: tuple[int, ...]  # continuous nTrode ids, in channel order
    channels: tuple[int, ...]  # DAQ channels
    period_ns: int


def file_streams(index: DaqIndex) -> list[DaqStream]:
    header = index.header
    boards = channel_boards(header)
    by_board: dict[tuple[int, int], list[int]] = {}
    for nt in sorted((n for n in header.ntrodes if n.id in index.pieces), key=lambda n: n.first_chan):
        by_board.setdefault((boards[nt.first_chan], index.pieces[nt.id].period_ns), []).append(nt.id)
    streams = []
    for (board, period), ids in sorted(by_board.items()):
        name = header.boards[board] if board < len(header.boards) and header.boards[board] else f"board{board + 1}"
        if sum(1 for b, _ in by_board if b == board) > 1:
            name += f" {1e9 / period:g} Hz"
        channels = tuple(c for i in ids for c in index.pieces[i].channels)
        streams.append(DaqStream(name, board, tuple(ids), channels, period))
    return streams


@dataclass(frozen=True)
class _RunPart:
    """Samples of one stream in one run: per nTrode the index of its first sample."""

    first_samples: tuple[int, ...]
    n_samples: int


class DaqFile:
    """A DAQ file with its runs (contiguous stretches of the first stream) and streams."""

    def __init__(self, path: str):
        self.index = read_index(path)
        self.streams = file_streams(self.index)
        if not self.streams:
            raise DaqFormatError("no continuous channels")
        self.fit = board_fit(self.index.events) if len({s.board for s in self.streams}) > 1 else None
        main = self.streams[0]
        self.regions = self.index.pieces[main.ntrodes[0]].regions()
        self._parts: dict[tuple[int, str], _RunPart] = {}
        # stream -> DAQ channel -> (position of its nTrode in the stream, column in the nTrode)
        self._columns = {s.name: {c: (k, column) for k, nid in enumerate(s.ntrodes)
                                  for column, c in enumerate(self.index.pieces[nid].channels)}
                         for s in self.streams}

    def to_main_time(self, board: int, times: np.ndarray | float) -> np.ndarray:
        """Times of a board's clock on the clock of the first board (ns)."""
        times = np.asarray(times, dtype=float)
        if board == self.streams[0].board or self.fit is None:
            return times
        return (times - self.fit["offset_ns"]) / self.fit["rate"]

    def from_main_time(self, board: int, times: np.ndarray | float) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        if board == self.streams[0].board or self.fit is None:
            return times
        return self.fit["offset_ns"] + self.fit["rate"] * times

    def run_window(self, region: int) -> tuple[int, int]:
        """Start and end of a run (ns, clock of the first board)."""
        _, n, t0 = self.regions[region]
        return t0, t0 + n * self.streams[0].period_ns

    def stream(self, name: str) -> DaqStream:
        return next(s for s in self.streams if s.name == name)

    def part(self, region: int, stream_name: str) -> _RunPart:
        """Samples of a stream inside run ``region``: within the run's time window, mapped
        onto the stream's board clock; the nTrodes of the stream start at the same time."""
        if (region, stream_name) not in self._parts:
            self._parts[region, stream_name] = self._find_part(region, stream_name)
        return self._parts[region, stream_name]

    def _find_part(self, region: int, stream_name: str) -> _RunPart:
        stream = self.stream(stream_name)
        t0, t1 = (float(x) for x in self.from_main_time(stream.board, np.array(self.run_window(region))))
        firsts, ends = [], []
        for nid in stream.ntrodes:
            pieces = self.index.pieces[nid]
            containing = [(a, n, t) for a, n, t in pieces.regions() if t <= t1 and t + n * pieces.period_ns > t0]
            if not containing:
                firsts.append(0)
                ends.append(0)
                continue
            a, n, _ = containing[0]
            first = int(np.ceil(np.clip(float(pieces.sample_at(t0)), a, a + n) - 1e-6))
            last = int(np.floor(np.clip(float(pieces.sample_at(t1)), a, a + n) + 1e-6))
            firsts.append(first)
            ends.append(last)
        return _RunPart(tuple(firsts), max(min(e - f for f, e in zip(firsts, ends)), 0))

    def read(self, region: int, stream_name: str, start: int, stop: int, channels: list[int]) -> np.ndarray:
        """Samples ``[start, stop)`` of DAQ ``channels`` of a stream in a run."""
        stream = self.stream(stream_name)
        part = self.part(region, stream_name)
        stop = min(stop, part.n_samples)
        out = np.empty((max(stop - start, 0), len(channels)), dtype=np.int16)
        wanted: dict[int, list[tuple[int, int]]] = {}  # nTrode position -> (output column, nTrode column)
        for out_column, channel in enumerate(channels):
            k, column = self._columns[stream_name][channel]
            wanted.setdefault(k, []).append((out_column, column))
        buf = _memmap(str(self.index.path))
        for k, pairs in wanted.items():
            first = part.first_samples[k]
            pieces = self.index.pieces[stream.ntrodes[k]]
            out[:, [o for o, _ in pairs]] = pieces.read(buf, first + start, first + stop, [c for _, c in pairs])
        return out

    def piece_times(self, region: int, stream_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Samples of a stream in a run at its first sample, at every piece start and at its
        end, and their times on the first board's clock (ns): the points of its clock map."""
        stream = self.stream(stream_name)
        part = self.part(region, stream_name)
        pieces = self.index.pieces[stream.ntrodes[0]]
        first = part.first_samples[0]
        inside = pieces.starts[(pieces.starts > first) & (pieces.starts < first + part.n_samples)]
        absolute = np.r_[first, inside, first + part.n_samples].astype(np.int64)
        # a sample at a piece start belongs to that piece; the end continues the last piece
        times = pieces.time_at(absolute)
        times[-1] = pieces.time_at(absolute[-1:] - 1)[0] + pieces.period_ns if part.n_samples else times[-1]
        return (absolute - first).astype(float), self.to_main_time(stream.board, times)


def daq_file(path: str) -> DaqFile:
    """The runs and streams of a DAQ file (cached like :func:`read_index`)."""
    return _daq_file(path, _version(path))


@cache
def _daq_file(path: str, version: tuple[int, int]) -> DaqFile:
    return DaqFile(path)


class DaqRecordingSegment(BaseRecordingSegment):
    def __init__(self, path: str, region: int, stream_name: str, channels: list[int], sampling_frequency: float):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._file = daq_file(path)
        self._region, self._stream, self._channels = region, stream_name, channels

    def get_num_samples(self) -> int:
        return self._file.part(self._region, self._stream).n_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        start, stop, _ = slice(start_frame, end_frame).indices(self.get_num_samples())
        channels = self._channels if channel_indices is None else list(np.asarray(self._channels)[channel_indices])
        return self._file.read(self._region, self._stream, start, stop, channels)


class DaqRecordingExtractor(BaseRecording):
    """One stream (board) of one run of a DAQ file: int16 ADC counts, gains in µV per count at
    the ADC input."""

    def __init__(self, path: str, region: int, stream_name: str):
        file = daq_file(path)
        stream = file.stream(stream_name)
        rate = 1e9 / stream.period_ns
        BaseRecording.__init__(self, rate, [f"AI{c}" for c in stream.channels], np.dtype("int16"))
        self.add_recording_segment(DaqRecordingSegment(path, region, stream_name, list(stream.channels), rate))
        channels = file.index.header.channels
        self.set_channel_gains([channels[c].volts_per_count * 1e6 for c in stream.channels])
        self.set_channel_offsets(0.0)
        self._kwargs = {"path": path, "region": region, "stream_name": stream_name}


@cache
def _extractor(path: str, version: tuple[int, int], region: int, stream_name: str) -> DaqRecordingExtractor:
    return DaqRecordingExtractor(path, region, stream_name)


# --- format -----------------------------------------------------------------------------------


def _daq_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".daq")


class DaqFormat(InputFormat):
    name: ClassVar[str] = "daq"

    def can_read(self, session_path: Path) -> bool:
        return any(p.suffix.lower() == ".daq" for p in session_path.iterdir() if p.is_file())

    def open_session(self, session_path: Path) -> Session:
        found = []
        for path in _daq_files(session_path):
            try:
                file = daq_file(str(path))
            except (DaqFormatError, struct.error) as error:
                logger.warning("%s: not read (%s)", path.name, error)
                continue
            found.append((file.index.header.start or datetime.max, path.name, path, file))
        runs: list[Run] = []
        for start, _, path, file in sorted(found, key=lambda item: item[:2]):
            index = file.index
            first_t0 = file.regions[0][2]
            for region, (_, n, t0) in enumerate(file.regions):
                main = file.streams[0]
                runs.append(Run(
                    name=path.stem if len(file.regions) == 1 else f"{path.stem}_part{region + 1}",
                    path=session_path,
                    index=len(runs),
                    format=self,
                    start_time=(start + timedelta(microseconds=(t0 - first_t0) / 1000)) if start != datetime.max else None,
                    duration_s=file.part(region, main.name).n_samples * main.period_ns / 1e9,
                    aux_stream=main.name,
                    streams=[s.name for s in file.streams],
                    clock_start_s=t0 / 1e9,  # the DAQ clock (time axis of a file per run)
                    extra={
                        "file": str(path),
                        "region": region,
                        "file_version": ".".join(map(str, index.header.file_version)),
                        "boards": list(index.header.boards),
                        "aux_streams": [main.name],
                        "start_ns": t0,
                        "board_fit": file.fit,
                        "damaged": list(index.damaged),
                        "truncated_bytes": index.truncated_bytes,
                        "spike_ntrodes": [nt.id for nt in index.header.ntrodes if nt.spike],
                    },
                ))
        return Session(name=session_path.name, path=session_path, format=self, runs=runs)

    # --- helpers ---

    def _file(self, run: Run) -> DaqFile:
        return daq_file(run.extra["file"])

    def _aux_sample(self, run: Run, times_ns: np.ndarray) -> np.ndarray:
        """Samples of the aux stream (first board) at times of the first board's clock."""
        file = self._file(run)
        main = file.streams[0]
        first = file.part(run.extra["region"], main.name).first_samples[0]
        return file.index.pieces[main.ntrodes[0]].sample_at(times_ns) - first

    def _events_in_run(self, run: Run) -> tuple[np.ndarray, np.ndarray]:
        """Event times on the first board's clock (ns) and lines (±) inside the run."""
        file = self._file(run)
        events = file.index.events
        times = events[:, 0].astype(float)
        lines = events[:, 1]
        if file.fit is not None:
            copied = np.abs(lines) > COPY_OFFSET
            times[copied] = file.to_main_time(1, times[copied])
        t0, t1 = file.run_window(run.extra["region"])
        inside = (times >= t0) & (times < t1)
        return times[inside], lines[inside]

    # --- InputFormat ---

    def default_aux_channels(self, run: Run | None = None) -> AuxChannelMap:
        from ephyscon.session_config import AuxChannelMap

        return AuxChannelMap(analog={}, digital={})  # the roles of the lines differ between setups

    def get_recording(self, run: Run, kind: StreamKind, probe_id: str | None = None) -> BaseRecording:
        if kind == StreamKind.AUX:
            stream_name = run.aux_stream
        else:
            if probe_id is None:
                raise ValueError(f"probe_id is required for {kind} streams")
            try:
                stream_name = run.probes[probe_id].streams[kind]
            except KeyError:
                raise KeyError(f"Run {run.name} has no {kind} stream for {probe_id}") from None
        if stream_name is None:
            raise KeyError(f"Run {run.name} has no auxiliary stream")
        return _extractor(run.extra["file"], _version(run.extra["file"]), run.extra["region"], stream_name)

    def aux_analog_recording(self, run: Run) -> BaseRecording:
        """The aux stream's channels that belong to no probe."""
        recording = self.get_recording(run, StreamKind.AUX)
        used = {c for info in run.probes.values() if info.streams.get(StreamKind.AP) == run.aux_stream
                for c in (info.channel_ids or recording.channel_ids)}
        return recording.select_channels([c for c in recording.channel_ids if c not in used])

    def clock_maps(self, run: Run, reference: str | None = None) -> RunSync | None:
        from ephyscon.sync import AUX, ClockMap, RunSync

        file = self._file(run)
        region = run.extra["region"]
        streams = {pid: info.streams[StreamKind.AP] for pid, info in run.probes.items() if StreamKind.AP in info.streams}
        streams[AUX] = file.streams[0].name
        t0 = file.run_window(region)[0]
        maps = {}
        for key, name in streams.items():
            samples, times = file.piece_times(region, name)
            maps[key] = ClockMap(samples, (times - t0) / 1e9, 1e9 / file.stream(name).period_ns)
        return RunSync(reference=reference or next(iter(streams)), maps=maps)

    def read_aux_analog(self, run: Run, index: int) -> np.ndarray:
        recording = self.get_recording(run, StreamKind.AUX)
        return recording.get_traces(channel_ids=[f"AI{index}"], return_in_uV=False)[:, 0]

    def aux_analog_gain_v(self, run: Run, index: int) -> float:
        return self._file(run).index.header.channels[index].volts_per_count

    def read_ttl_events(self, run: Run, lines: Collection[int] | None = None) -> pd.DataFrame:
        times, signed = self._events_in_run(run)
        result = pd.DataFrame({"aux_sample": np.floor(self._aux_sample(run, times) + 1e-6).astype(np.int64),
                               "line": np.abs(signed).astype(np.int64), "state": (signed > 0).astype(np.int64)})
        return keep_lines(result.sort_values(["aux_sample", "line"], kind="stable", ignore_index=True), lines)

    def read_aux_digital(self, run: Run, bit: int) -> np.ndarray:
        n = self.get_recording(run, StreamKind.AUX).get_num_samples()
        line = np.zeros(n, dtype=bool)
        events = self.read_ttl_events(run)
        mine = events[events["line"] == bit + 1]
        level, previous = False, 0
        for sample, state in zip(np.clip(mine["aux_sample"].to_numpy(), 0, n), mine["state"].to_numpy()):
            line[previous:sample] = level
            level, previous = bool(state), int(sample)
        line[previous:] = level
        return line

    def read_messages(self, run: Run) -> pd.DataFrame:
        return pd.DataFrame({"aux_sample": np.zeros(0, np.int64), "text": np.zeros(0, str)})

    def read_trial_descriptors(self, run: Run) -> pd.DataFrame:
        """``TD01`` records of the run (written by the stimulus PC): ``aux_sample``,
        ``trial_number``, ``stimulus_number``."""
        file = self._file(run)
        trials = file.index.trials
        t0, t1 = file.run_window(run.extra["region"])
        inside = (trials[:, 0] >= t0) & (trials[:, 0] < t1)
        return pd.DataFrame({"aux_sample": np.floor(self._aux_sample(run, trials[inside, 0]) + 1e-6).astype(np.int64),
                             "trial_number": trials[inside, 1], "stimulus_number": trials[inside, 2]})
