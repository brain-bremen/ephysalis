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

from ephyscon.plugins import INPUT_FORMAT_GROUP, available, get_plugin
from ephyscon.session import Run, Session, StreamKind

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording


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
