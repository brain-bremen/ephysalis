"""Format-independent description of a recording session.

A *session* is one experimental day/penetration (e.g. ``SOMxCD1_172_20250109_Recording1``).
It consists of one or more *runs* (contiguous acquisitions, e.g. one SpikeGLX gate folder
``3_rfmappingFine_g0`` or one Open Ephys recording). Each run contains one or more *probes*
and optionally an *auxiliary* stream (NIDQ, Open Ephys ADC/NI-DAQ) carrying sync, photodiode,
wheel and laser signals. Spike sorting concatenates the runs of a session per probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

    from ephysalis.io.base import InputFormat


class StreamKind(StrEnum):
    AP = "ap"  # wideband / action potential band
    LF = "lf"  # hardware low-pass band (Neuropixels 1.0 only)
    AUX = "aux"  # analog/digital auxiliary channels (NIDQ, Open Ephys ADC)
    SYNC = "sync"  # probe sync channel


@dataclass(frozen=True)
class ProbeInfo:
    probe_id: str  # stable key across runs, e.g. "imec0"
    model_name: str | None = None  # e.g. "Neuropixels 1.0 probe with cap"
    part_number: str | None = None  # e.g. "PRB_1_4_0480_1_C"
    serial_number: str | None = None
    manufacturer: str | None = None  # e.g. "IMEC"
    num_shanks: int = 1
    num_channels: int = 0
    has_lf: bool = False
    streams: dict[StreamKind, str] = field(default_factory=dict)  # kind -> format stream id
    # probes defined in session.toml (ephysalis.io.probes): channels of the stream, geometry
    channel_ids: tuple[str, ...] | None = None
    geometry: dict | None = None
    # calibration (ephysalis.calibration): factors on the recorded gains of the probe's channels
    # (AP and LF streams), NaN for channels without calibration
    gain_factors: tuple[float, ...] | None = None


@dataclass
class Run:
    name: str
    path: Path
    index: int  # 0-based position within the session (acquisition order)
    format: InputFormat
    start_time: datetime | None = None
    duration_s: float | None = None
    probes: dict[str, ProbeInfo] = field(default_factory=dict)
    aux_stream: str | None = None
    streams: list[str] = field(default_factory=list)  # names of all streams (probes in session.toml pick one)
    # time of the first sample on the acquisition clock (s), the time axis of a file per run
    # (``export --per-run``); 0 for formats whose clock starts with the recording
    clock_start_s: float = 0.0
    # start time as recorded, when ``start_time`` was replaced by a better one (ephysalis.runstart)
    recorded_start_time: datetime | None = None
    start_time_method: str = ""  # how ``start_time`` was obtained, if not from the recording
    extra: dict = field(default_factory=dict)  # format-specific details, used only by the format
    def get_recording(self, kind: StreamKind, probe_id: str | None = None) -> BaseRecording:
        recording = self.format.get_recording(self, kind, probe_id)
        info = self.probes.get(probe_id) if probe_id is not None else None
        if info is not None and kind in (StreamKind.AP, StreamKind.LF) and (info.channel_ids or info.geometry):
            from ephysalis.io.probes import apply_probe

            recording = apply_probe(recording, info)
        if info is not None and kind in (StreamKind.AP, StreamKind.LF) and info.gain_factors is not None:
            gains = recording.get_channel_gains() * np.asarray(info.gain_factors)
            if info.channel_ids is None and not info.geometry:
                recording = recording.select_channels(list(recording.channel_ids))  # a copy: readers cache theirs
            recording.set_channel_gains(list(gains))
        return recording


@dataclass
class Session:
    name: str
    path: Path
    format: InputFormat
    runs: list[Run]

    @property
    def probe_ids(self) -> list[str]:
        ids: list[str] = []
        for run in self.runs:
            for pid in run.probes:
                if pid not in ids:
                    ids.append(pid)
        return ids

    def run(self, name: str) -> Run:
        for run in self.runs:
            if run.name == name:
                return run
        raise KeyError(f"No run named {name!r} in session {self.name!r}")
