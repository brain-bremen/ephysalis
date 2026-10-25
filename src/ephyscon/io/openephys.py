"""Open Ephys reader for the three recording formats of the Open Ephys GUI: Binary
(``structure.oebin``), NWB (``experiment<N>.nwb``) and the legacy Open Ephys format
(``*.continuous``), built on `open-ephys-python-tools <https://github.com/open-ephys/open-ephys-python-tools>`_.

An Open Ephys session folder contains ``Record Node <id>`` folders (or is one). Each recording
of the first record node (further nodes typically hold copies of the same data) becomes one
:class:`~ephyscon.session.Run` ``experiment<N>_recording<M>``. open-ephys-python-tools detects
the format and the recordings and reads stream metadata, TTL events and messages. Samples are
read lazily by :class:`OpenEphysRecordingExtractor`: from the package's memory maps (Binary)
and HDF5 datasets (NWB); in the legacy format, whose loader copies whole channels into memory,
the 1024-sample records of the ``.continuous`` files are memory-mapped directly.

Streams are named ``<record node>#<processor>-<id>.<stream>`` (the Binary folder names).
Neuropixels streams from the Neuropix-PXI / OneBox plugins (``...ProbeA-AP``, ``...ProbeA-LFP``
or ``...ProbeA`` for 2.0 probes) are mapped to probes, with the probe geometry from
``settings.xml``; all other continuous streams (NI-DAQmx, acquisition-board ADCs) are
auxiliary streams. As in neo, SYNC channels are left out and the ADC channels of a stream
with neural channels form a separate stream ``<stream>_ADC``.

Open Ephys synchronises all streams of a recording itself (timestamps of each stream, in
seconds on the main stream's clock), so the clock maps come from these timestamps instead of
from sync pulses (:meth:`OpenEphysFormat.clock_maps`).

Auxiliary signals: analog input ``n`` is the aux stream's channel ``AI<n>`` (NI-DAQmx) or
``ADC<n+1>`` (acquisition board); digital bit ``b`` is TTL line ``b + 1`` of the aux stream
(Open Ephys numbers lines from 1). Network Events words (VStim) are the states of the
plugin's TTL lines after the last event of each sample, which is what the Binary format
stores as ``full_words``. Messages come from the Message Center.

Run start times come from the "Software Time" sync message of each recording (ms since
1970 UTC, Open Ephys >= 0.6), or else from ``<DATE>`` in ``settings.xml`` (start of the
experiment, 1 s resolution). They are naive local times, like those of SpikeGLX.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from ephyscon.io.base import InputFormat
from ephyscon.session import ProbeInfo, Run, Session, StreamKind

if TYPE_CHECKING:
    from open_ephys.analysis.recording import Recording as OERecording

    from ephyscon.sync import RunSync

_PROBE_STREAM_RE = re.compile(r"(?P<probe>Probe[A-Z]+)(?:-(?P<band>AP|LFP))?$")
_SOFTWARE_TIME_RE = re.compile(r"Software Time.*?:\s*(?P<ms>\d+)")
_SETTINGS_DATE_RE = re.compile(r"<DATE>(?P<date>[^<]+)</DATE>")
_ADC_SUFFIX = "_ADC"
NETWORK_EVENTS_PROCESSOR = "Network Events"


@dataclass(frozen=True)
class OEStream:
    """One continuous stream of a recording (part of a stored stream, see the module docstring)."""

    name: str  # "<record node>#<processor>-<id>.<stream>[_ADC]"
    index: int  # position of the stored stream in the recording
    processor_id: int
    oe_stream: str  # stream name in the Open Ephys GUI (events are labelled with it)
    sample_rate: float
    columns: tuple[int, ...]  # columns of the stored data
    channel_names: tuple[str, ...]
    bit_volts: tuple[float, ...]
    units: tuple[str, ...]  # "uV" or "V"

    @property
    def folder(self) -> str:
        return self.name.split("#", 1)[-1]


def _same_stream(a: str, b: str) -> bool:
    """Stream names of events and continuous data differ in punctuation in the legacy format
    (``example-data`` vs ``example_data``)."""
    return re.sub(r"[^0-9a-z]", "", a.lower()) == re.sub(r"[^0-9a-z]", "", b.lower())


def _folder(processor_name: str, processor_id: int, oe_stream: str) -> str:
    """Binary-format folder name of a stream, e.g. ``NI-DAQmx-110.USB-6343``."""
    return f"{processor_name.replace(' ', '_')}-{processor_id}.{oe_stream}"


def _split_stream(node: str, folder: str, index: int, processor_id: int, oe_stream: str,
                  sample_rate: float, names: list[str], bit_volts: list[float], units: list[str]) -> list[OEStream]:
    """Neural and ADC channels of a stored stream; SYNC channels are left out."""
    base = f"{node}#{folder}"
    units = [u or ("V" if "ADC" in n else "uV") for n, u in zip(names, units)]
    keep = [i for i, n in enumerate(names) if "SYNC" not in n]
    adc = [i for i in keep if names[i].startswith("ADC")]
    neural = [i for i in keep if i not in adc]
    parts = [(base, neural), (base + _ADC_SUFFIX, adc)] if neural and adc else [(base, keep)]
    return [OEStream(name, index, processor_id, oe_stream, float(sample_rate), tuple(cols),
                     tuple(names[i] for i in cols), tuple(float(bit_volts[i]) for i in cols),
                     tuple(units[i] for i in cols))
            for name, cols in parts if cols]


def _streams(recording: OERecording, node: str) -> list[OEStream]:
    fmt = str(recording.format)
    streams: list[OEStream] = []
    if fmt == "binary":
        for index, info in enumerate(recording.info["continuous"]):  # type: ignore[attr-defined]
            channels = info["channels"]
            streams += _split_stream(node, info["folder_name"].rstrip("/"), index, int(info["source_processor_id"]),
                                     info["stream_name"],
                                     info["sample_rate"], [c["channel_name"] for c in channels],
                                     [c["bit_volts"] for c in channels], [c.get("units", "") for c in channels])
    elif fmt == "nwb":
        for index, continuous in enumerate(recording.continuous.values()):  # type: ignore[union-attr]
            meta = continuous.metadata
            n = meta.num_channels
            # NWB stores no channel names; conversion is volts per bit (the package returns uV per bit)
            folder = _folder(meta.source_node_name, int(meta.source_node_id), meta.stream_name)
            streams += _split_stream(node, folder, index, int(meta.source_node_id), meta.stream_name, meta.sample_rate, [f"CH{i + 1}" for i in range(n)], list(meta.bit_volts),
                                     ["uV"] * n)
    else:
        _, _, _, infos = recording.find_continuous_files()  # type: ignore[attr-defined]
        for index, info in enumerate(infos):
            names = info["channel_names"]
            folder = _folder(info["source_node_name"], int(info["source_node_id"]), info["stream_name"])
            streams += _split_stream(node, folder, index, int(info["source_node_id"]), info["stream_name"],
                                     info["sample_rate"], names, info["bit_volts"],
                                     [""] * len(names))
    return streams


# --- samples -------------------------------------------------------------------------------

_LEGACY_RECORD = np.dtype([("sample_number", "<i8"), ("n", "<u2"), ("recording", "<u2"),
                           ("samples", ">i2", (1024,)), ("marker", "u1", (10,))])
_LEGACY_HEADER_BYTES = 1024


class _LegacyChannels:
    """Samples of one legacy-format stream: one ``.continuous`` file per channel."""

    def __init__(self, files: list[str], recording_index: int):
        self.records = []
        for file in files:
            data = np.memmap(file, dtype=_LEGACY_RECORD, mode="r", offset=_LEGACY_HEADER_BYTES)
            self.records.append(data[np.flatnonzero(data["recording"] == recording_index)])
        first = self.records[0]
        self.n_samples = int(first["n"].astype(np.int64).sum())
        self.first_sample = int(first["sample_number"][0]) if len(first) else 0

    def __getitem__(self, key: tuple[slice, list[int]]) -> np.ndarray:
        rows, columns = key
        start, stop, _ = rows.indices(self.n_samples)
        first, last = start // 1024, max(start, stop - 1) // 1024 + 1
        out = np.empty((max(stop - start, 0), len(columns)), dtype=np.int16)
        for k, column in enumerate(columns):
            block = self.records[column]["samples"][first:last].reshape(-1)
            out[:, k] = block[start - first * 1024: stop - first * 1024]
        return out


@dataclass
class _StoredData:
    samples: Any  # (n_samples, n_columns), sliceable like an array (memmap, HDF5 dataset, _LegacyChannels)
    n_samples: int
    sample_numbers: Any  # sample number of each sample (array-like)
    timestamps: Any  # seconds on the main clock (array-like)


@cache
def _oe_recordings(record_node: str) -> tuple[OERecording, ...]:
    from open_ephys.analysis.recordnode import RecordNode

    return tuple(RecordNode(record_node).recordings)


@cache
def _stored_data(record_node: str, position: int, index: int) -> _StoredData:
    recording = _oe_recordings(record_node)[position]
    if str(recording.format) == "open-ephys":
        files, stream_indexes, _, infos = recording.find_continuous_files()  # type: ignore[attr-defined]
        directory = Path(recording.directory)
        channels = _LegacyChannels([str(directory / f) for f, s in zip(files, stream_indexes) if s == index],
                                   recording.recording_index)
        numbers = np.arange(channels.first_sample, channels.first_sample + channels.n_samples)
        return _StoredData(channels, channels.n_samples, numbers, numbers / float(infos[index]["sample_rate"]))
    continuous = recording.continuous[index]  # type: ignore[index]
    return _StoredData(continuous.samples, int(continuous.samples.shape[0]), continuous.sample_numbers,
                       continuous.timestamps)


class OpenEphysRecordingSegment(BaseRecordingSegment):
    def __init__(self, data: _StoredData, columns: tuple[int, ...], sampling_frequency: float):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._data = data
        self._columns = list(columns)

    def get_num_samples(self) -> int:
        return self._data.n_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        columns = self._columns if channel_indices is None else list(np.asarray(self._columns)[channel_indices])
        rows = slice(start_frame, end_frame)
        if isinstance(self._data.samples, _LegacyChannels):
            return self._data.samples[rows, columns]
        return np.asarray(self._data.samples[rows])[:, columns]


class OpenEphysRecordingExtractor(BaseRecording):
    """One stream of an Open Ephys recording (any of the three formats), int16 with gains in
    µV per bit; Neuropixels streams carry their probe from ``settings.xml``."""

    def __init__(self, record_node: str, position: int, stream_name: str):
        stream = next(s for s in _node_streams(record_node, position) if s.name == stream_name)
        BaseRecording.__init__(self, stream.sample_rate, list(stream.channel_names), np.dtype("int16"))
        self.add_recording_segment(
            OpenEphysRecordingSegment(_stored_data(record_node, position, stream.index), stream.columns,
                                      stream.sample_rate))
        to_uv = [1e6 if unit == "V" else 1.0 for unit in stream.units]
        self.set_channel_gains([b * f for b, f in zip(stream.bit_volts, to_uv)])
        self.set_channel_offsets(0.0)
        self.set_property("gain_to_physical_unit", list(stream.bit_volts))
        self.set_property("physical_unit", list(stream.units))
        if _PROBE_STREAM_RE.search(stream.folder):
            self._attach_probe(record_node, position, stream)
        self._kwargs = {"record_node": record_node, "position": position, "stream_name": stream_name}

    def _attach_probe(self, record_node: str, position: int, stream: OEStream) -> None:
        import probeinterface
        from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts_from_probe

        settings = _settings_file(record_node, _oe_recordings(record_node)[position].experiment_index)
        if settings is None or not probeinterface.has_neuropixels_probes(settings, stream_name=stream.folder):
            return
        probe = probeinterface.read_openephys_neuropixels(settings_file=settings, stream_name=stream.folder)
        if probe.shank_ids is not None:
            self.set_probe(probe, group_mode="by_shank")
        else:
            self.set_probe(probe)
        shifts = get_neuropixels_sample_shifts_from_probe(probe)
        if shifts is not None:
            self.set_property("inter_sample_shift", shifts)


@cache
def _node_streams(record_node: str, position: int) -> tuple[OEStream, ...]:
    return tuple(_streams(_oe_recordings(record_node)[position], Path(record_node).name))


@cache
def _extractor(record_node: str, position: int, stream_name: str) -> OpenEphysRecordingExtractor:
    return OpenEphysRecordingExtractor(record_node, position, stream_name)


# --- metadata --------------------------------------------------------------------------------


def _settings_file(record_node: str | Path, experiment_index: int) -> Path | None:
    path = Path(record_node) / ("settings.xml" if experiment_index == 0 else f"settings_{experiment_index + 1}.xml")
    return path if path.exists() else None


@cache
def _processor_names(settings: Path | None) -> dict[int, str]:
    if settings is None:
        return {}
    root = ElementTree.parse(settings).getroot()
    return {int(p.get("nodeId", -1)): p.get("name", "") for p in root.iter("PROCESSOR")}


def recording_start_time(recording: OERecording) -> datetime | None:
    """Start of a recording (naive local time), see the module docstring."""
    fmt = str(recording.format)
    directory = Path(recording.directory)
    software_ms = None
    if fmt == "binary" and (directory / "sync_messages.txt").exists():
        match = _SOFTWARE_TIME_RE.search((directory / "sync_messages.txt").read_text(encoding="utf-8", errors="replace"))
        software_ms = int(match["ms"]) if match else None
    elif fmt == "nwb" and "sync_messages" in recording.nwb["acquisition"]:  # type: ignore[attr-defined]
        group = recording.nwb["acquisition"]["sync_messages"]  # type: ignore[attr-defined]
        for text, value in zip(group["data"][()], group["sync"][()]):
            if b"Software Time" in bytes(text):
                software_ms = int(value)
                break
    elif fmt == "open-ephys":
        suffix = "" if recording.experiment_index == 0 else f"_{recording.experiment_index + 1}"
        messages = directory / f"messages{suffix}.events"
        if messages.exists():
            times = [int(line.split(",", 1)[0]) for line in messages.read_text(encoding="utf-8", errors="replace").splitlines()
                     if "Software Time" in line]
            if recording.recording_index < len(times):
                software_ms = times[recording.recording_index]
    if software_ms is not None:
        return datetime.fromtimestamp(software_ms / 1000)
    root = directory if fmt != "binary" else directory.parent.parent
    settings = _settings_file(root, recording.experiment_index)
    if settings is not None:
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
        if session_path.name.startswith("Record Node"):
            record_node = session_path
        else:
            # Only use the first record node; further nodes typically hold copies of the same data.
            from open_ephys.analysis.utils import alphanum_key

            record_node = sorted(session_path.glob("Record Node *"), key=lambda p: alphanum_key(p.name))[0]
        node = str(record_node)
        runs: list[Run] = []
        for position, recording in enumerate(_oe_recordings(node)):
            streams = _node_streams(node, position)
            probes: dict[str, dict[StreamKind, str]] = defaultdict(dict)
            aux_streams = []
            for stream in streams:
                match = _PROBE_STREAM_RE.search(stream.folder)
                if match is None:
                    aux_streams.append(stream.name)
                else:
                    probes[match["probe"]][StreamKind.LF if match["band"] == "LFP" else StreamKind.AP] = stream.name
            # analog-input streams (NI-DAQmx, ADC channels) before other non-probe streams
            analog = [s.name for s in streams if s.name in aux_streams
                      and all(re.match(r"(AI|ADC)\d+$", c) for c in s.channel_names)]
            aux_streams = analog + [s for s in aux_streams if s not in analog]
            main = next((s for s in streams if s.name in (aux_streams[:1] or [streams[0].name])), None) if streams else None
            duration = _stored_data(node, position, main.index).n_samples / main.sample_rate if main else None
            runs.append(
                Run(
                    name=f"experiment{recording.experiment_index + 1}_recording{recording.recording_index + 1}",
                    path=session_path,
                    index=len(runs),
                    format=self,
                    start_time=recording_start_time(recording),
                    duration_s=duration,
                    probes={pid: ProbeInfo(probe_id=pid, has_lf=StreamKind.LF in kinds, streams=dict(kinds))
                            for pid, kinds in sorted(probes.items())},
                    aux_stream=aux_streams[0] if aux_streams else None,
                    extra={
                        "position": position,
                        "oe_format": str(recording.format),
                        "aux_streams": aux_streams,
                        "streams": [s.name for s in streams],
                        "record_node": record_node,
                    },
                )
            )
        return Session(name=session_path.name, path=session_path, format=self, runs=runs)

    # --- helpers ---

    def _node(self, run: Run) -> str:
        return str(run.extra["record_node"])

    def _oe_recording(self, run: Run) -> OERecording:
        return _oe_recordings(self._node(run))[run.extra["position"]]

    def _stream(self, run: Run, stream_name: str) -> OEStream:
        return next(s for s in _node_streams(self._node(run), run.extra["position"]) if s.name == stream_name)

    def _data(self, run: Run, stream_name: str) -> _StoredData:
        return _stored_data(self._node(run), run.extra["position"], self._stream(run, stream_name).index)

    def _aux(self, run: Run) -> OEStream:
        if run.aux_stream is None:
            raise KeyError(f"Run {run.name} has no auxiliary stream")
        return self._stream(run, run.aux_stream)

    def _to_aux_samples(self, run: Run, events: pd.DataFrame) -> np.ndarray:
        """Aux-stream sample of each event (columns ``sample_number``, ``stream_name``,
        ``timestamp``): sample numbers of events of the aux stream itself, otherwise the time."""
        aux = self._aux(run)
        data = self._data(run, aux.name)
        first = int(np.asarray(data.sample_numbers[:1])[0]) if data.n_samples else 0
        samples = events["sample_number"].to_numpy(np.int64) - first
        if "timestamp" in events and len(events):
            other = ~events["stream_name"].map(lambda s: _same_stream(str(s), aux.oe_stream)).to_numpy(bool)
            if other.any():
                samples[other] = np.searchsorted(np.asarray(data.timestamps), events["timestamp"].to_numpy(float)[other])
        return samples

    def _events(self, run: Run) -> pd.DataFrame:
        """TTL events of all processors; Binary in file order (see :meth:`read_network_events`),
        the other formats sorted by sample number (stable)."""
        recording = self._oe_recording(run)
        if hasattr(recording, "sort_events"):
            recording.sort_events = False
        events = recording.events
        if events is None or not len(events):
            return pd.DataFrame(columns=["line", "sample_number", "timestamp", "processor_id", "stream_name", "state"])
        return events.astype({"processor_id": np.int64, "line": np.int64, "state": np.int64})

    # --- InputFormat ---

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
            timestamps[key] = (self._data(run, name).timestamps, self._stream(run, name).sample_rate)
        t0 = float(np.asarray(timestamps[reference][0][:1])[0])
        maps = {}
        for key, (ts, rate) in timestamps.items():
            step = max(1, int(rate))  # one point per second, plus the last sample
            samples = np.unique(np.r_[np.arange(0, len(ts), step), len(ts) - 1])
            maps[key] = ClockMap(samples.astype(float), np.asarray(ts[samples], dtype=float) - t0, rate)
        return RunSync(reference=reference, maps=maps)

    def _aux_channel_id(self, run: Run, index: int) -> str:
        channel_ids = [str(c) for c in self.get_recording(run, StreamKind.AUX).channel_ids]
        for candidate in (f"AI{index}", f"ADC{index + 1}"):
            if candidate in channel_ids:
                return candidate
        if run.extra.get("oe_format") == "nwb" and index < len(channel_ids):  # NWB has no channel names
            return channel_ids[index]
        raise IndexError(f"Run {run.name}: no analog input {index} (AI{index}/ADC{index + 1}) in {channel_ids}")

    def read_aux_analog(self, run: Run, index: int) -> np.ndarray:
        recording = self.get_recording(run, StreamKind.AUX)
        return recording.get_traces(channel_ids=[self._aux_channel_id(run, index)], return_in_uV=False)[:, 0]

    def aux_analog_gain_v(self, run: Run, index: int) -> float:
        recording = self.get_recording(run, StreamKind.AUX)
        position = [str(c) for c in recording.channel_ids].index(self._aux_channel_id(run, index))
        return float(recording.get_channel_gains()[position]) * 1e-6  # spikeinterface gains are uV per bit

    def read_aux_digital(self, run: Run, bit: int) -> np.ndarray:
        n = self.get_recording(run, StreamKind.AUX).get_num_samples()
        line = np.zeros(n, dtype=bool)
        events = self.read_ttl_events(run)
        mine = events[events["line"] == bit + 1]
        level = False
        previous = 0
        for sample, state in zip(np.clip(mine["aux_sample"].to_numpy(), 0, n), mine["state"].to_numpy()):
            line[previous:sample] = level
            level, previous = bool(state), int(sample)
        line[previous:] = level
        return line

    def read_ttl_events(self, run: Run) -> pd.DataFrame:
        aux = self._aux(run)
        events = self._events(run)
        mine = events[(events["processor_id"] == aux.processor_id)
                      & events["stream_name"].map(lambda s: _same_stream(str(s), aux.oe_stream)).astype(bool)]
        result = pd.DataFrame({"aux_sample": self._to_aux_samples(run, mine), "line": mine["line"].to_numpy(np.int64),
                               "state": mine["state"].to_numpy(np.int64)})
        return result.sort_values(["aux_sample", "line"], kind="stable", ignore_index=True)

    def read_network_events(self, run: Run) -> pd.DataFrame:
        """Words of the Network Events plugin: the plugin sets the lines of a word one by one in
        the same sample, so the word is the line state after the last event of each sample.
        The state is followed in the order the plugin wrote the events: words sent in quick
        succession can be written out of time order, and their lines then interleave
        differently than in time order. Only the Binary format keeps that order (open-ephys-python-tools
        sorts NWB and legacy events by sample number)."""
        empty = super().read_network_events(run)
        if run.aux_stream is None:
            return empty
        names = _processor_names(_settings_file(self._node(run), self._oe_recording(run).experiment_index))
        events = self._events(run)
        events = events[events["processor_id"].map(lambda p: names.get(int(p)) == NETWORK_EVENTS_PROCESSOR).astype(bool)]
        if not len(events):
            return empty
        words = np.empty(len(events), dtype=np.int64)
        word = 0
        for k, (line, state) in enumerate(zip(events["line"].tolist(), events["state"].tolist())):
            bit = 1 << ((line - 1) % 32)
            word = word | bit if state else word & ~bit
            words[k] = word
        numbers = events["sample_number"].to_numpy(np.int64)
        last = np.r_[numbers[1:] != numbers[:-1], True]  # last event of each sample (run of equal samples)
        result = pd.DataFrame({"aux_sample": self._to_aux_samples(run, events[last]), "word": words[last]})
        return result.sort_values("aux_sample", kind="stable", ignore_index=True)

    def read_messages(self, run: Run) -> pd.DataFrame:
        empty = pd.DataFrame({"aux_sample": np.zeros(0, np.int64), "text": np.zeros(0, str)})
        if run.aux_stream is None:
            return empty
        recording = self._oe_recording(run)
        if run.extra.get("oe_format") == "nwb":  # open-ephys-python-tools does not read NWB messages yet
            acquisition = recording.nwb["acquisition"]  # type: ignore[attr-defined]
            if "messages" not in acquisition:
                return empty
            group = acquisition["messages"]
            messages = pd.DataFrame({"sample_number": group["sync"][()], "timestamp": group["timestamps"][()],
                                     "message": [bytes(m).decode("utf-8", errors="replace") for m in group["data"][()]]})
        else:
            messages = recording.messages
            if messages is None or not len(messages):
                return empty
            if run.extra.get("oe_format") == "open-ephys":  # sample numbers in the "timestamp" column
                messages = pd.DataFrame({"sample_number": messages["timestamp"].to_numpy(np.int64),
                                         "message": messages["message"].str.strip().to_numpy()})
        if "timestamp" in messages:  # on the main clock: place by time
            data = self._data(run, run.aux_stream)
            samples = np.searchsorted(np.asarray(data.timestamps), messages["timestamp"].to_numpy(float))
        else:
            samples = self._to_aux_samples(run, messages.assign(stream_name=self._aux(run).oe_stream))
        return pd.DataFrame({"aux_sample": np.asarray(samples, dtype=np.int64), "text": messages["message"].to_numpy()})

    def get_recording(
        self, run: Run, kind: StreamKind, probe_id: str | None = None
    ) -> BaseRecording:
        if kind == StreamKind.AUX:
            stream_name = self._aux(run).name
        else:
            if probe_id is None:
                raise ValueError(f"probe_id is required for {kind} streams")
            try:
                stream_name = run.probes[probe_id].streams[kind]
            except KeyError:
                raise KeyError(f"Run {run.name} has no {kind} stream for {probe_id}") from None
        return _extractor(self._node(run), run.extra["position"], stream_name)
