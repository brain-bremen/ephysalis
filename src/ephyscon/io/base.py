"""Interface for acquisition-system readers.

Readers turn a session folder into a :class:`~ephyscon.session.Session` and hand out lazy
spikeinterface recordings for each stream. Everything downstream (sync, sorting, LFP,
writers) only talks to this interface, so adding a new acquisition system (e.g. Multi
Channel Systems) means adding one reader and registering it under the
``ephyscon.input_formats`` entry point group.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from ephyscon.plugins import INPUT_FORMAT_GROUP, available, get_plugin
from ephyscon.session import Run, Session, StreamKind

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

    from ephyscon.session_config import AuxChannelMap
    from ephyscon.sync import RunSync, SyncSource


class InputFormat(ABC):
    name: ClassVar[str]

    @abstractmethod
    def can_read(self, session_path: Path) -> bool:
        """Cheap check whether ``session_path`` looks like a session of this format."""

    @abstractmethod
    def open_session(self, session_path: Path) -> Session:
        """Discover all runs, probes and streams of a session (metadata only, no data loaded)."""

    @abstractmethod
    def get_recording(
        self, run: Run, kind: StreamKind, probe_id: str | None = None
    ) -> BaseRecording:
        """Return a lazy recording of one stream of a run."""

    def default_aux_channels(self, run: Run | None = None) -> AuxChannelMap:
        """Roles of the aux channels when ``session.toml`` defines none (default: the MATLAB
        pipeline's 8-input NIDQ layout)."""
        from ephyscon.session_config import AuxChannelMap

        return AuxChannelMap.legacy_default(8)

    # Sync and auxiliary channels. Optional: formats without them can't be synchronised.

    def clock_maps(self, run: Run, reference: str | None = None) -> RunSync | None:
        """Clock maps of all streams from timestamps the format already synchronised (e.g. Open
        Ephys ``timestamps.npy``), or ``None`` if streams must be synchronised from sync
        pulses (:meth:`sync_source`)."""
        return None

    def sync_source(self, run: Run, probe_id: str | None, aux_channels: AuxChannelMap | None) -> SyncSource:
        """Channel and bit carrying the sync pulses of a probe, or of the aux stream (``None``)."""
        raise NotImplementedError(f"{self.name} does not support sync yet")

    def aux_sampling_frequency(self, run: Run) -> float:
        return self.get_recording(run, StreamKind.AUX).sampling_frequency

    def read_aux_analog(self, run: Run, index: int) -> np.ndarray:
        """Raw ADC counts of analog input ``index`` of the aux stream (full run)."""
        raise NotImplementedError(f"{self.name} does not support aux analog channels yet")

    def aux_analog_gain_v(self, run: Run, index: int) -> float:
        """Volts per ADC count of analog input ``index``."""
        raise NotImplementedError(f"{self.name} does not support aux analog channels yet")

    def read_messages(self, run: Run) -> pd.DataFrame:
        """Text messages of a run (e.g. Open Ephys Message Center, sent by VStim):
        ``aux_sample`` (sample of the aux stream at the message time) and ``text``."""
        raise NotImplementedError(f"{self.name} has no messages")

    def read_aux_digital(self, run: Run, bit: int) -> np.ndarray:
        """One digital line (bit of the digital word) of the aux stream as a bool array."""
        raise NotImplementedError(f"{self.name} does not support aux digital lines yet")

    def aux_analog_recording(self, run: Run) -> BaseRecording:
        """Lazy recording of the analog inputs of the aux stream (gains in µV per bit), e.g.
        for the raw data in DH5 / NWB."""
        return self.get_recording(run, StreamKind.AUX)

    def read_ttl_events(self, run: Run) -> pd.DataFrame:
        """Edges of all digital lines of the aux stream: ``aux_sample``, ``line`` (1-based,
        bit + 1) and ``state`` (1 rising, 0 falling), in time order."""
        raise NotImplementedError(f"{self.name} does not support TTL events yet")

    def read_network_events(self, run: Run) -> pd.DataFrame:
        """Network event words of a run (e.g. Open Ephys Network Events, sent by VStim):
        ``aux_sample`` and ``word``. Empty for formats without network events."""
        return pd.DataFrame({"aux_sample": np.zeros(0, np.int64), "word": np.zeros(0, np.int64)})


def open_session(session_path: str | Path, format_name: str | None = None) -> Session:
    """Open a session, auto-detecting the input format unless ``format_name`` is given."""
    session_path = Path(session_path)
    if not session_path.is_dir():
        raise NotADirectoryError(session_path)

    if format_name is not None:
        return get_plugin(INPUT_FORMAT_GROUP, format_name).open_session(session_path)

    for name in available(INPUT_FORMAT_GROUP):
        reader: InputFormat = get_plugin(INPUT_FORMAT_GROUP, name)
        if reader.can_read(session_path):
            return reader.open_session(session_path)
    raise ValueError(
        f"Could not detect input format of {session_path}. "
        f"Available formats: {available(INPUT_FORMAT_GROUP)}"
    )
