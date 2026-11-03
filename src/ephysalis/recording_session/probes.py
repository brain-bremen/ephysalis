"""Probes defined in ``session.toml`` for streams a reader does not recognise as probes.

Readers detect Neuropixels probes themselves. Any other probe, e.g. a 32-channel probe on
the Open Ephys acquisition board (Intan headstage), is defined in ``[probes.<id>]``::

    [probes.probe0]
    stream = "Acquisition_Board-100.Rhythm Data"   # input stream (short or full name)
    channels = ["CH1", "CH2", ..., "CH32"]          # default: all channels of the stream
    geometry = { manufacturer = "cambridgeneurotech", model = "ASSY-37-E-1", wiring = "ASSY-156>RHD2164" }
    # or: geometry = { file = "my_probe.json" }     # probeinterface JSON

The geometry comes from the probeinterface library (downloaded once) or a probeinterface
file; ``wiring`` maps the probe's contacts to the headstage channels (probeinterface
``wiring_to_device``). Without ``wiring``, ``channels`` are in contact order.

``reference_channels`` re-references the probe: the mean of these channels of the same stream
(e.g. a local reference electrode) is subtracted from the probe's channels in every step::

    [probes.v4]
    stream = "elec0003"
    channels = ["18C", "20C"]
    reference_channels = ["14C"]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from ephysalis.plugins import ListsStreamChannelIds
from ephysalis.recording_session.model import ProbeInfo, RecordingSession, StreamKind

if TYPE_CHECKING:
    from probeinterface import Probe

    from ephysalis.recording_session.config import ProbeGeometry, RecordingSessionConfig

logger = logging.getLogger(__name__)


def _match_stream(names: list[str], wanted: str) -> str | None:
    for name in names:
        if wanted in (name, name.split("#", 1)[-1]):
            return name
    matches = [n for n in names if n.endswith(wanted)]
    return matches[0] if len(matches) == 1 else None


def configure_probes(recording_session: RecordingSession, config: RecordingSessionConfig | None, config_dir: Path | None = None) -> RecordingSession:
    """Add the probes defined in ``config`` (``stream`` set) to every run that has the stream and,
    when the format can tell (:meth:`InputFormat.stream_channel_ids`), records all of the probe's
    channels and reference channels; other runs get no such probe (a warning)."""
    if config is None:
        return recording_session
    for probe_id, probe_config in config.probes.items():
        if probe_config.stream is None:
            continue
        geometry = probe_config.geometry.model_dump(exclude_none=True) if probe_config.geometry else None
        if geometry and "file" in geometry and config_dir is not None and not Path(geometry["file"]).is_absolute():
            geometry["file"] = str(config_dir / geometry["file"])
        found = False
        for run in recording_session.runs:
            stream = _match_stream(run.streams, probe_config.stream)
            if stream is None:
                continue
            wanted = [*(probe_config.channels or []), *(probe_config.reference_channels or [])]
            recorded = (run.format.list_stream_channel_ids(run, stream)
                        if wanted and isinstance(run.format, ListsStreamChannelIds) else None)
            if recorded is not None and (missing := [c for c in wanted if c not in set(recorded)]):
                logger.warning("probe %s left out of run %s: channels %s not recorded in stream %s",
                               probe_id, run.name, missing, stream)
                continue
            found = True
            run.probes[probe_id] = ProbeInfo(
                probe_id=probe_id,
                model_name=geometry.get("model") if geometry else None,
                part_number=geometry.get("model") if geometry else None,
                serial_number=probe_config.serial_number,
                manufacturer=geometry.get("manufacturer") if geometry else None,
                num_channels=len(probe_config.channels or []),
                streams={StreamKind.AP: stream},
                channel_ids=tuple(probe_config.channels) if probe_config.channels else None,
                geometry=geometry,
                reference_channel_ids=(tuple(probe_config.reference_channels)
                                       if probe_config.reference_channels else None),
            )
        if not found:
            logger.warning("probe %s: no run has a stream %r", probe_id, probe_config.stream)
    return recording_session


def build_probe(geometry: dict) -> Probe:
    import probeinterface

    if "file" in geometry:
        probe = probeinterface.read_probeinterface(geometry["file"]).probes[0]
    else:
        probe = probeinterface.get_probe(geometry["manufacturer"], geometry["model"])
    if "wiring" in geometry:
        probe.wiring_to_device(geometry["wiring"])
    return probe


class _RereferencedSegment(BaseRecordingSegment):
    def __init__(self, parent: Any, sampling_frequency: float, channels: np.ndarray, references: np.ndarray):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency, t_start=getattr(parent, "t_start", None))
        self._parent, self._channels, self._references = parent, channels, references

    def get_num_samples(self) -> int:
        return self._parent.get_num_samples()

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        channels = self._channels if channel_indices is None else self._channels[channel_indices]
        wanted = np.unique(np.r_[channels, self._references])
        traces = self._parent.get_traces(start_frame, end_frame, wanted).astype(np.float32)
        column = {c: k for k, c in enumerate(wanted)}
        reference = traces[:, [column[c] for c in self._references]].mean(axis=1, keepdims=True)
        return traces[:, [column[c] for c in channels]] - reference


class RereferencedRecording(BaseRecording):
    """``channel_ids`` of ``recording`` minus the mean of its ``reference_ids`` (float32, in the recording's counts;
    the gains of all channels involved must be equal)."""

    def __init__(self, recording: BaseRecording, channel_ids: list, reference_ids: list):
        all_ids = list(recording.channel_ids)
        missing = [c for c in [*channel_ids, *reference_ids] if c not in all_ids]
        if missing:
            raise ValueError(f"re-reference: channels {missing} are not in the stream")
        gains = recording.get_channel_gains()
        involved = [all_ids.index(c) for c in [*channel_ids, *reference_ids]]
        if not np.allclose(gains[involved], gains[involved[0]]):
            raise ValueError("re-reference: the channels and the reference channels have different gains")
        BaseRecording.__init__(self, recording.sampling_frequency, list(channel_ids), np.dtype("float32"))
        channels = np.array([all_ids.index(c) for c in channel_ids])
        references = np.array([all_ids.index(c) for c in reference_ids])
        for segment in recording._recording_segments:
            self.add_recording_segment(_RereferencedSegment(segment, recording.sampling_frequency, channels, references))
        self.set_channel_gains(gains[channels])
        self.set_channel_offsets(0.0)
        for key in recording.get_property_keys():
            values = recording.get_property(key)
            if values is not None and len(values) == len(all_ids):
                self.set_property(key, np.asarray(values)[channels])
        self.annotate(reference_channels=list(reference_ids))
        self._kwargs = {"recording": recording, "channel_ids": list(channel_ids), "reference_ids": list(reference_ids)}


def rereference(recording: BaseRecording, channel_ids: list, reference_ids: list) -> BaseRecording:
    """The probe's channels minus the mean of the reference channels (``probes.<id>.reference_channels``)."""
    return RereferencedRecording(recording, channel_ids, reference_ids)


def apply_probe(recording: BaseRecording, info: ProbeInfo) -> BaseRecording:
    """Select the probe's channels of its stream and attach its geometry."""
    # always a new (channel-selected) recording: set_probe changes it in place, and readers cache theirs
    channel_ids = list(info.channel_ids) if info.channel_ids is not None else list(recording.channel_ids)
    if info.reference_channel_ids:
        recording = rereference(recording, channel_ids, list(info.reference_channel_ids))
    else:
        recording = recording.select_channels(channel_ids)
    if info.geometry is None:
        return recording
    probe = build_probe(info.geometry)
    if probe.get_contact_count() != recording.get_num_channels():
        raise ValueError(
            f"probe {info.probe_id}: {probe.get_contact_count()} contacts but {recording.get_num_channels()} channels"
        )
    if probe.device_channel_indices is None:
        probe.set_device_channel_indices(np.arange(probe.get_contact_count()))
    recording.set_probe(probe)
    return recording

