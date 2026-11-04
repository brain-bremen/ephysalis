"""Events of a run: edges of the digital lines and event words, on the reference clock.

Every run gets ``events/<run>.csv`` (written by ``ephysalis extract-events``):

==========  ===============================================================================
column      content
==========  ===============================================================================
``time``    s on the run's reference clock
``source``  ``ttl`` (digital line of the aux stream) or ``word`` (event word, e.g. Open Ephys
            network events; input formats with :class:`~ephysalis.plugins.ReadsEventWords`)
``line``    TTL line (1-based, i.e. bit + 1)
``state``   TTL line level after the edge (1 rising, 0 falling)
``word``    event word (e.g. a VStim event code)
``name``    role of the TTL line (``[aux_channels]``) or the word's name from a metadata plugin
            (:class:`~ephysalis.plugins.NamesEventWords`)
==========  ===============================================================================

Lines with ``in_events = false`` in ``[aux_channels.digital]`` (by default the sync and wheel lines, which
switch thousands of times per minute) are left out, with ``lines_without_role = false`` also
all lines without a role (e.g. unconnected inputs that toggle).

The table is independent of the output format: the DH5 event codes (``+line`` / ``-line`` for
rising / falling TTL edges, ``word + [events] event_word_offset`` for event words) are made
by the DH5 export (:func:`dh5_event_codes`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ephysalis.plugins import NamesEventWords, ReadsDigitalLineEdges, ReadsEventWords, metadata_plugins
from ephysalis.timeline.clocks import AUX

if TYPE_CHECKING:
    from ephysalis.recording_session.model import Run
    from ephysalis.recording_session.config import AuxChannelMap, EventsConfig
    from ephysalis.timeline.clocks import RunSync

logger = logging.getLogger(__name__)

COLUMNS = ["time", "source", "line", "state", "word", "name"]
INT_COLUMNS = ("line", "state", "word")


def event_word_names(run: Run) -> dict[int, str]:
    """Names of the run's event words: from the first metadata plugin that names them."""
    for plugin in metadata_plugins():
        if isinstance(plugin, NamesEventWords) and (names := plugin.name_event_words(run)):
            return names
    return {}


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float if c == "time" else object) for c in COLUMNS})


def run_events(run: Run, sync: RunSync, aux_channels: AuxChannelMap, config: EventsConfig) -> pd.DataFrame:
    """Digital line edges and event words of a run (see the module docs)."""
    if run.aux_stream is None:
        return empty_events()
    roles = {entry.bit + 1: role for role, entry in aux_channels.digital.items()}
    excluded = {entry.bit + 1 for entry in aux_channels.digital.values() if not entry.in_events}
    wanted = None if config.lines_without_role else set(roles) - excluded
    parts = []
    ttl = (run.format.read_digital_line_edges(run, wanted) if isinstance(run.format, ReadsDigitalLineEdges)
           else pd.DataFrame({"aux_sample": [], "line": [], "state": []}))
    ttl = ttl[~ttl.line.isin(excluded)]
    if len(ttl):
        line, state = ttl.line.to_numpy(int), ttl.state.to_numpy(int)
        parts.append(pd.DataFrame({
            "time": sync.to_reference_time(AUX, ttl.aux_sample.to_numpy(float)), "source": "ttl",
            "line": line, "state": state, "word": pd.NA,
            "name": [roles.get(n, "") for n in line],
        }))
    words = run.format.read_event_words(run) if isinstance(run.format, ReadsEventWords) else pd.DataFrame()
    if len(words):
        names = event_word_names(run)
        word = words.word.to_numpy(int)
        parts.append(pd.DataFrame({
            "time": sync.to_reference_time(AUX, words.aux_sample.to_numpy(float)), "source": "word",
            "line": pd.NA, "state": pd.NA, "word": word,
            "name": [names.get(int(w), "") for w in word],
        }))
    if not parts:
        return empty_events()
    events = pd.concat(parts, ignore_index=True).sort_values("time", kind="stable", ignore_index=True)
    for column in INT_COLUMNS:
        events[column] = events[column].astype("Int64")
    logger.info("Run %s: %d TTL edges, %d event words (left out lines %s)", run.name,
                (events.source == "ttl").sum(), (events.source == "word").sum(), sorted(excluded) or "none")
    return events[COLUMNS]


def load_run_events(path) -> pd.DataFrame:
    """``events/<run>.csv``."""
    events = pd.read_csv(path, keep_default_na=False, na_values={c: [""] for c in INT_COLUMNS})
    for column in INT_COLUMNS:
        events[column] = events[column].astype("Int64")
    return events


def dh5_event_codes(events: pd.DataFrame, event_word_offset: int) -> np.ndarray:
    """DH5 event codes: ``+line`` / ``-line`` for rising / falling TTL edges, ``word + offset``
    for event words."""
    ttl = (events.source == "ttl").to_numpy()
    line = events.line.fillna(0).to_numpy(np.int64)
    state = events.state.fillna(0).to_numpy(np.int64)
    word = events.word.fillna(0).to_numpy(np.int64)
    return np.where(ttl, np.where(state > 0, line, -line), word + event_word_offset).astype(np.int32)
