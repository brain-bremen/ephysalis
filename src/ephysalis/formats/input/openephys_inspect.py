"""Lightweight inspection of an Open Ephys session (no data loading)."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class StreamInfo:
    name: str
    num_channels: int
    sample_rate: float
    duration_s: float | None  # derived from .dat file size
    channel_names: list[str] = field(default_factory=list)


@dataclass
class EventStreamInfo:
    name: str
    count: int | None  # None if timestamps.npy could not be read
    message_breakdown: dict[str, int] | None = None  # For Message Center: counts by message type


@dataclass
class RecordingInfo:
    experiment_index: int  # 0-based
    recording_index: int   # 0-based
    directory: str
    streams: list[StreamInfo] = field(default_factory=list)
    event_streams: list[EventStreamInfo] = field(default_factory=list)

    @property
    def duration_s(self) -> float | None:
        """Duration of the first continuous stream, or None."""
        return self.streams[0].duration_s if self.streams else None

    @property
    def num_continuous_channels(self) -> int:
        return sum(s.num_channels for s in self.streams)

    @property
    def event_stream_names(self) -> list[str]:
        return [e.name for e in self.event_streams]


@dataclass
class SessionInfo:
    path: Path
    total_size_bytes: int
    recordings: list[RecordingInfo] = field(default_factory=list)

    @property
    def num_recordings(self) -> int:
        return len(self.recordings)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _folder_size(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fname))
            except OSError:
                pass
    return total


def _dat_duration(dat_path: str, num_channels: int, sample_rate: float) -> float | None:
    try:
        size = os.path.getsize(dat_path)
        num_samples = size // (num_channels * 2)  # int16 → 2 bytes
        return num_samples / sample_rate
    except OSError:
        return None


def _event_count(rec_dir: str, folder_name: str) -> int | None:
    ts_path = os.path.join(rec_dir, "events", folder_name, "timestamps.npy")
    try:
        return int(np.load(ts_path, mmap_mode="r").shape[0])
    except Exception:
        return None


def _count_message_types(rec_dir: str, folder_name: str) -> dict[str, int] | None:
    """Count Message Center messages by type (TRIAL_START, TRIAL_END, etc.)."""
    text_path = os.path.join(rec_dir, "events", folder_name, "text.npy")
    try:
        messages = np.load(text_path, mmap_mode="r")
        counts: dict[str, int] = {}

        for msg in messages:
            # Decode bytes to string
            msg_str = msg.decode() if isinstance(msg, bytes) else str(msg)

            # Count specific message types
            if "TRIAL_START" in msg_str:
                counts["TRIAL_START"] = counts.get("TRIAL_START", 0) + 1
            elif "TRIAL_END" in msg_str:
                counts["TRIAL_END"] = counts.get("TRIAL_END", 0) + 1
            else:
                counts["Other"] = counts.get("Other", 0) + 1

        return counts if counts else None
    except Exception:
        return None


def _inspect_recording(rec_dir: str, exp_idx: int, rec_idx: int) -> RecordingInfo:
    oebin_path = os.path.join(rec_dir, "structure.oebin")
    streams: list[StreamInfo] = []
    event_streams: list[EventStreamInfo] = []

    if os.path.isfile(oebin_path):
        with open(oebin_path) as f:
            info = json.load(f)
        for cont in info.get("continuous", []):
            name = cont.get("stream_name") or cont.get("folder_name", "?")
            num_ch = int(cont.get("num_channels", 0))
            sr = float(cont.get("sample_rate", 0))
            dat = os.path.join(rec_dir, "continuous", cont["folder_name"], "continuous.dat")
            dur = _dat_duration(dat, num_ch, sr) if num_ch and sr else None
            channel_names = [ch["channel_name"] for ch in cont.get("channels", []) if "channel_name" in ch]
            if not channel_names and num_ch:
                channel_names = [f"CH{i}" for i in range(num_ch)]
            streams.append(StreamInfo(name=name, num_channels=num_ch, sample_rate=sr, duration_s=dur, channel_names=channel_names))
        seen: set[str] = set()
        for ev in info.get("events", []):
            name = ev.get("source_processor") or ev.get("stream_name", "?")
            if name in seen:
                continue
            seen.add(name)
            count = _event_count(rec_dir, ev["folder_name"])
            # For Message Center events, also count message types
            breakdown = _count_message_types(rec_dir, ev["folder_name"]) if name == "Message Center" else None
            event_streams.append(EventStreamInfo(name=name, count=count, message_breakdown=breakdown))
    else:
        for d in sorted(glob.glob(os.path.join(rec_dir, "events", "*"))):
            if os.path.isdir(d):
                folder = os.path.basename(d)
                name = ".".join(folder.split(".")[1:]) or folder  # strip node prefix
                count = _event_count(rec_dir, folder)
                # For Message Center events, also count message types
                breakdown = _count_message_types(rec_dir, folder) if "Message Center" in name else None
                event_streams.append(EventStreamInfo(name=name, count=count, message_breakdown=breakdown))

    return RecordingInfo(
        experiment_index=exp_idx,
        recording_index=rec_idx,
        directory=rec_dir,
        streams=streams,
        event_streams=event_streams,
    )


def _find_recordings(node_dir: str) -> list[tuple[int, int, str]]:
    """Return (exp_idx, rec_idx, path) for all recordings in a record-node dir."""
    result = []
    exp_dirs = sorted(glob.glob(os.path.join(node_dir, "experiment*")))
    for exp_idx, exp_dir in enumerate(exp_dirs):
        rec_dirs = sorted(glob.glob(os.path.join(exp_dir, "recording*")))
        for rec_idx, rec_dir in enumerate(rec_dirs):
            result.append((exp_idx, rec_idx, rec_dir))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_session_path(session_path: Path) -> None:
    """Raise ValueError if *session_path* does not look like an Open Ephys session.

    Accepts either a full session folder (contains settings.xml) or a single
    experiment folder (contains recording* subdirectories).
    """
    if not session_path.exists():
        raise ValueError(f"Path does not exist: {session_path}")
    if not session_path.is_dir():
        raise ValueError(f"Not a directory: {session_path}")
    if bool(glob.glob(str(session_path / "Record Node *"))):
        return  # full session folder

    if bool(glob.glob(str(session_path / "recording*"))):
        # Could be an experiment folder — valid only if an ancestor contains a Record Node folder
        for parent in session_path.parents:
            if bool(glob.glob(str(parent / "Record Node *"))):
                return
        raise ValueError(
            f"'{session_path.name}' looks like an experiment folder but no Open Ephys "
            "session (Record Node folder) was found in any parent directory."
        )

    raise ValueError(
        f"'{session_path.name}' does not appear to be an Open Ephys session folder.\n"
        "Expected 'Record Node' subdirectories (full session) "
        "or recording* subdirectories inside a session (single experiment)."
    )


def inspect_session(session_path: Path) -> SessionInfo:
    """Return a :class:`SessionInfo` for the given Open Ephys session folder.

    This function only reads metadata (JSON + file sizes) — it does not
    memory-map or load any signal data.
    """
    size = _folder_size(session_path)
    recordings: list[RecordingInfo] = []

    node_dirs = sorted(glob.glob(str(session_path / "Record Node *")))
    if not node_dirs:
        node_dirs = [str(session_path)]

    for node_dir in node_dirs:
        for exp_idx, rec_idx, rec_dir in _find_recordings(node_dir):
            recordings.append(_inspect_recording(rec_dir, exp_idx, rec_idx))

    return SessionInfo(path=session_path, total_size_bytes=size, recordings=recordings)


def _fmt_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _fmt_duration(s: float | None) -> str:
    if s is None:
        return "?"
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _fmt_rate(hz: float) -> str:
    return f"{hz / 1000:.1f} kHz" if hz >= 1000 else f"{hz:.0f} Hz"


def format_session_info(info: SessionInfo) -> str:
    """Return a human-readable summary string of *info*."""
    lines: list[str] = []
    lines.append(f"Session : {info.path.name}")
    lines.append(f"Size    : {_fmt_size(info.total_size_bytes)}")
    lines.append(f"Recordings: {info.num_recordings}")

    for rec in info.recordings:
        lines.append(
            f"\n  Exp {rec.experiment_index + 1} / Rec {rec.recording_index + 1}"
            f"  [{_fmt_duration(rec.duration_s)}]"
        )
        if rec.streams:
            stream_parts = [
                f"{s.name} ({s.num_channels} ch, {_fmt_rate(s.sample_rate)})"
                for s in rec.streams
            ]
            lines.append(f"    Continuous : {',  '.join(stream_parts)}")
        else:
            lines.append("    Continuous : —")
        if rec.event_streams:
            ev_parts = [
                f"{e.name} ({e.count})" if e.count is not None else e.name
                for e in rec.event_streams
            ]
            lines.append(f"    Events     : {',  '.join(ev_parts)}")
        else:
            lines.append("    Events     : —")

    return "\n".join(lines)
