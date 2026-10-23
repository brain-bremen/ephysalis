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

from ephyscon.plugins import INPUT_FORMAT_GROUP, available, get_plugin
from ephyscon.session import Run, Session, StreamKind

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

    from ephyscon.session_config import AuxChannelMap
    from ephyscon.sync import Edges


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

    # Sync and auxiliary channels. Optional: formats without them can't be synchronised.

    def sync_edges(self, run: Run, probe_id: str | None, aux_channels: AuxChannelMap | None) -> Edges:
        """Edges of the sync pulse train recorded by a probe, or by the aux stream (``None``)."""
        raise NotImplementedError(f"{self.name} does not support sync edges yet")

    def aux_sampling_frequency(self, run: Run) -> float:
        return self.get_recording(run, StreamKind.AUX).sampling_frequency

    def read_aux_analog(self, run: Run, index: int) -> np.ndarray:
        """Raw ADC counts of analog input ``index`` of the aux stream (full run)."""
        raise NotImplementedError(f"{self.name} does not support aux analog channels yet")

    def aux_analog_gain_v(self, run: Run, index: int) -> float:
        """Volts per ADC count of analog input ``index``."""
        raise NotImplementedError(f"{self.name} does not support aux analog channels yet")

    def read_aux_digital(self, run: Run, bit: int) -> np.ndarray:
        """One digital line (bit of the digital word) of the aux stream as a bool array."""
        raise NotImplementedError(f"{self.name} does not support aux digital lines yet")


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
