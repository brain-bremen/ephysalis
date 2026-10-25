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
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ephyscon.session import ProbeInfo, Session, StreamKind

if TYPE_CHECKING:
    from probeinterface import Probe
    from spikeinterface.core import BaseRecording

    from ephyscon.session_config import ProbeGeometry, SessionConfig

logger = logging.getLogger(__name__)


def _match_stream(names: list[str], wanted: str) -> str | None:
    for name in names:
        if wanted in (name, name.split("#", 1)[-1]):
            return name
    matches = [n for n in names if n.endswith(wanted)]
    return matches[0] if len(matches) == 1 else None


def configure_probes(session: Session, config: SessionConfig | None, config_dir: Path | None = None) -> Session:
    """Add the probes defined in ``config`` (``stream`` set) to every run that has the stream."""
    if config is None:
        return session
    for probe_id, probe_config in config.probes.items():
        if probe_config.stream is None:
            continue
        geometry = probe_config.geometry.model_dump(exclude_none=True) if probe_config.geometry else None
        if geometry and "file" in geometry and config_dir is not None and not Path(geometry["file"]).is_absolute():
            geometry["file"] = str(config_dir / geometry["file"])
        found = False
        for run in session.runs:
            stream = _match_stream(run.extra.get("streams", []), probe_config.stream)
            if stream is None:
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
            )
        if not found:
            logger.warning("probe %s: no run has a stream %r", probe_id, probe_config.stream)
    return session


def build_probe(geometry: dict) -> Probe:
    import probeinterface

    if "file" in geometry:
        probe = probeinterface.read_probeinterface(geometry["file"]).probes[0]
    else:
        probe = probeinterface.get_probe(geometry["manufacturer"], geometry["model"])
    if "wiring" in geometry:
        probe.wiring_to_device(geometry["wiring"])
    return probe


def apply_probe(recording: BaseRecording, info: ProbeInfo) -> BaseRecording:
    """Select the probe's channels of its stream and attach its geometry."""
    # always a new (channel-selected) recording: set_probe changes it in place, and readers cache theirs
    channel_ids = list(info.channel_ids) if info.channel_ids is not None else list(recording.channel_ids)
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

