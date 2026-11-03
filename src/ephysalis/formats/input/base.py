"""Opening a session with its input format.

Input formats are plugins (:class:`ephysalis.plugins.InputFormat` and its optional capabilities,
entry point group ``ephysalis.input_formats``): typically a module of functions that turns a
session folder into a :class:`~ephysalis.session.model.Session` and hands out lazy spikeinterface
recordings for each stream. Everything downstream (clock alignment, sorting, LFP, writers) only
talks to that interface.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import pandas as pd

from ephysalis.plugins import INPUT_FORMAT_GROUP, InputFormat, available, get_plugin
from ephysalis.session.model import Session


def keep_lines(events: pd.DataFrame, lines: Collection[int] | None) -> pd.DataFrame:
    """The digital line edges of ``lines`` (all if None)."""
    if lines is None:
        return events
    return events[events["line"].isin(list(lines))].reset_index(drop=True)


def open_session(session_path: str | Path, format_name: str | None = None) -> Session:
    """Open a session, detecting the input format unless ``format_name`` is given."""
    session_path = Path(session_path)
    if not session_path.is_dir():
        raise NotADirectoryError(session_path)
    if format_name is not None:
        return get_plugin(INPUT_FORMAT_GROUP, format_name).open_session(session_path)
    for name in available(INPUT_FORMAT_GROUP):
        reader: InputFormat = get_plugin(INPUT_FORMAT_GROUP, name)
        if reader.can_read_session_folder(session_path):
            return reader.open_session(session_path)
    raise ValueError(f"Could not detect the input format of {session_path}. Available formats: "
                     f"{available(INPUT_FORMAT_GROUP)}")
