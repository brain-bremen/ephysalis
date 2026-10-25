"""Open Ephys (binary format) reader.

An Open Ephys session folder contains ``Record Node <id>/experiment<N>/recording<M>``.
Each (experiment, recording) pair becomes one :class:`~ephyscon.session.Run`. Neuropixels
streams from the Neuropix-PXI / OneBox plugins (``...ProbeA-AP``, ``...ProbeA-LFP`` or
``...ProbeA`` for 2.0 probes) are mapped to probes; all other continuous streams
(NI-DAQmx, acquisition-board ADCs) are treated as auxiliary streams.

Open Ephys synchronises all streams of a recording itself (``timestamps.npy`` of each stream,
in seconds on the main stream's clock), so the clock maps come from these timestamps instead
of from sync pulses (:meth:`OpenEphysFormat.clock_maps`).

Auxiliary signals: analog input ``n`` is the aux stream's channel ``AI<n>`` (NI-DAQmx) or
``ADC<n+1>`` (acquisition board); digital bit ``b`` is TTL line ``b + 1`` of the aux stream's
``events/<stream>/TTL`` (Open Ephys numbers lines from 1). Messages come from
``events/MessageCenter``.

Run start times come from ``sync_messages.txt`` of each recording ("Software Time", ms since
1970 UTC, written by Open Ephys >= 0.6), or else from ``<DATE>`` in ``settings.xml`` (start of
the experiment, 1 s resolution). They are naive local times, like those of SpikeGLX.

Event (TTL / message) handling lives in the sync/stimulus layers and still comes from
oecon (to be ported).
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from ephyscon.io.base import InputFormat
from ephyscon.session import ProbeInfo, Run, Session, StreamKind

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

    from ephyscon.sync import RunSync

_PROBE_STREAM_RE = re.compile(r"(?P<probe>Probe[A-Z]+)(?:-(?P<band>AP|LFP))?$")
_SOFTWARE_TIME_RE = re.compile(r"Software Time.*?:\s*(?P<ms>\d+)")
_SETTINGS_DATE_RE = re.compile(r"<DATE>(?P<date>[^<]+)</DATE>")


def recording_start_time(record_node: Path, block_index: int, segment_index: int) -> datetime | None:
    """Start of ``experiment<block+1>/recording<segment+1>`` of a record node (naive local time)."""
    sync_messages = record_node / f"experiment{block_index + 1}" / f"recording{segment_index + 1}" / "sync_messages.txt"
    if sync_messages.exists():
        match = _SOFTWARE_TIME_RE.search(sync_messages.read_text(encoding="utf-8", errors="replace"))
        if match:
            return datetime.fromtimestamp(int(match["ms"]) / 1000)
    settings = record_node / ("settings.xml" if block_index == 0 else f"settings_{block_index + 1}.xml")
    if settings.exists():
        match = _SETTINGS_DATE_RE.search(settings.read_text(encoding="utf-8", errors="replace"))
        if match:
            return datetime.strptime(match["date"].strip(), "%d %b %Y %H:%M:%S")
    return None


class OpenEphysFormat(InputFormat):
    name: ClassVar[str] = "openephys"

    def can_read(self, session_path: Path) -> bool:
        return any(session_path.glob("Record Node *")) or session_path.name.startswith(
            "Record Node"
        )

    def open_session(self, session_path: Path) -> Session:
        import neo

        rawio = neo.rawio.OpenEphysBinaryRawIO(str(session_path))
        rawio.parse_header()
        stream_names = list(rawio.header["signal_streams"]["name"])

        # Only use the first record node; further nodes typically hold copies of the same data.
        record_nodes = sorted({name.split("#")[0] for name in stream_names if "#" in name})
        if record_nodes:
            stream_names = [n for n in stream_names if n.startswith(record_nodes[0] + "#")]

        probes: dict[str, dict[StreamKind, str]] = defaultdict(dict)
        aux_streams = []
        for full_name in stream_names:
            short = full_name.split("#", 1)[-1]
            match = _PROBE_STREAM_RE.search(short)
            if match is None:
                aux_streams.append(full_name)
                continue
            band = match["band"]
            probes[match["probe"]][StreamKind.LF if band == "LFP" else StreamKind.AP] = full_name

        if session_path.name.startswith("Record Node"):
            record_node = session_path
        else:
            record_node = session_path / (record_nodes[0] if record_nodes else next(session_path.glob("Record Node *")).name)

        runs: list[Run] = []
        for block in range(rawio.block_count()):
            for segment in range(rawio.segment_count(block)):
                t_start = rawio.segment_t_start(block, segment)
                t_stop = rawio.segment_t_stop(block, segment)
                probe_infos = {
                    pid: ProbeInfo(
                        probe_id=pid, has_lf=StreamKind.LF in kinds, streams=dict(kinds)
                    )
                    for pid, kinds in sorted(probes.items())
                }
                runs.append(
                    Run(
                        name=f"experiment{block + 1}_recording{segment + 1}",
                        path=session_path,
                        index=len(runs),
                        format=self,
                        start_time=recording_start_time(record_node, block, segment),
                        duration_s=t_stop - t_start,
                        probes=probe_infos,
                        aux_stream=aux_streams[0] if aux_streams else None,
                        extra={
                            "block_index": block,
                            "segment_index": segment,
                            "aux_streams": aux_streams,
                            "streams": stream_names,
                            "record_node": record_node,
                        },
                    )
                )
        return Session(name=session_path.name, path=session_path, format=self, runs=runs)

    def stream_folder(self, run: Run, stream_name: str) -> Path:
        node = run.extra["record_node"]
        return (node / f"experiment{run.extra['block_index'] + 1}" / f"recording{run.extra['segment_index'] + 1}"
                / "continuous" / stream_name.split("#", 1)[-1])

    def clock_maps(self, run: Run, reference: str | None = None) -> RunSync | None:
        from ephyscon.sync import AUX, ClockMap, RunSync

        streams = {pid: info.streams[StreamKind.AP] for pid, info in run.probes.items() if StreamKind.AP in info.streams}
        if run.aux_stream:
            streams[AUX] = run.aux_stream
        if not streams:
            return None
        reference = reference or next(iter(streams))
        timestamps = {}
        for key, name in streams.items():
            path = self.stream_folder(run, name) / "timestamps.npy"
            if not path.exists():
                return None
            timestamps[key] = np.load(path, mmap_mode="r")
        t0 = float(timestamps[reference][0])
        maps = {}
        for key, ts in timestamps.items():
            rate = self.get_recording(run, StreamKind.AUX if key == AUX else StreamKind.AP, None if key == AUX else key).sampling_frequency
            step = max(1, int(rate))  # one point per second, plus the last sample
            samples = np.unique(np.r_[np.arange(0, len(ts), step), len(ts) - 1])
            maps[key] = ClockMap(samples.astype(float), np.asarray(ts[samples], dtype=float) - t0, rate)
        return RunSync(reference=reference, maps=maps)

    def _recording_folder(self, run: Run) -> Path:
        node = run.extra["record_node"]
        return node / f"experiment{run.extra['block_index'] + 1}" / f"recording{run.extra['segment_index'] + 1}"

    def _aux_channel_id(self, run: Run, index: int) -> str:
        channel_ids = [str(c) for c in self.get_recording(run, StreamKind.AUX).channel_ids]
        for candidate in (f"AI{index}", f"ADC{index + 1}"):
            if candidate in channel_ids:
                return candidate
        raise IndexError(f"Run {run.name}: no analog input {index} (AI{index}/ADC{index + 1}) in {channel_ids}")

    def read_aux_analog(self, run: Run, index: int) -> np.ndarray:
        recording = self.get_recording(run, StreamKind.AUX)
        return recording.get_traces(channel_ids=[self._aux_channel_id(run, index)], return_in_uV=False)[:, 0]

    def aux_analog_gain_v(self, run: Run, index: int) -> float:
        recording = self.get_recording(run, StreamKind.AUX)
        position = [str(c) for c in recording.channel_ids].index(self._aux_channel_id(run, index))
        return float(recording.get_channel_gains()[position]) * 1e-6  # spikeinterface gains are uV per bit

    def read_aux_digital(self, run: Run, bit: int) -> np.ndarray:
        if run.aux_stream is None:
            raise KeyError(f"Run {run.name} has no auxiliary stream")
        folder = self.stream_folder(run, run.aux_stream)
        ttl = self._recording_folder(run) / "events" / folder.name / "TTL"
        first_sample = int(np.load(folder / "sample_numbers.npy", mmap_mode="r")[0])
        n = self.get_recording(run, StreamKind.AUX).get_num_samples()
        line = np.zeros(n, dtype=bool)
        if not ttl.exists():
            return line
        states = np.load(ttl / "states.npy")
        samples = np.load(ttl / "sample_numbers.npy").astype(np.int64) - first_sample
        mine = np.abs(states) == bit + 1
        level = False
        previous = 0
        for sample, state in zip(np.clip(samples[mine], 0, n), states[mine]):
            line[previous:sample] = level
            level, previous = bool(state > 0), int(sample)
        line[previous:] = level
        return line

    def read_messages(self, run: Run) -> pd.DataFrame:
        folder = self._recording_folder(run) / "events" / "MessageCenter"
        if not (folder / "text.npy").exists() or run.aux_stream is None:
            return pd.DataFrame({"aux_sample": np.zeros(0, np.int64), "text": np.zeros(0, str)})
        text = [m.decode("utf-8", errors="replace") if isinstance(m, bytes) else str(m) for m in np.load(folder / "text.npy")]
        times = np.load(folder / "timestamps.npy")
        aux_times = np.load(self.stream_folder(run, run.aux_stream) / "timestamps.npy", mmap_mode="r")
        return pd.DataFrame({"aux_sample": np.searchsorted(aux_times, times).astype(np.int64), "text": text})

    def get_recording(
        self, run: Run, kind: StreamKind, probe_id: str | None = None
    ) -> BaseRecording:
        if kind == StreamKind.AUX:
            if run.aux_stream is None:
                raise KeyError(f"Run {run.name} has no auxiliary stream")
            stream_name = run.aux_stream
        else:
            if probe_id is None:
                raise ValueError(f"probe_id is required for {kind} streams")
            try:
                stream_name = run.probes[probe_id].streams[kind]
            except KeyError:
                raise KeyError(f"Run {run.name} has no {kind} stream for {probe_id}") from None
        recording = _read_openephys(str(run.path), stream_name, run.extra["block_index"])
        return recording.select_segments([run.extra["segment_index"]])


@cache
def _read_openephys(folder: str, stream_name: str, block_index: int) -> BaseRecording:
    import spikeinterface.extractors as se

    return se.read_openephys(folder, stream_name=stream_name, block_index=block_index)
