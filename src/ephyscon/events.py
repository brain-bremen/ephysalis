"""Events of a run: TTL lines of the aux stream and VStim network events, on the reference clock.

Every run gets ``events/<run>.csv`` (written by ``ephyscon timing``):

==========  ===============================================================================
column      content
==========  ===============================================================================
``time``    s on the run's reference clock
``source``  ``ttl`` (digital line of the aux stream) or ``network`` (Open Ephys Network Events)
``line``    TTL line (1-based, i.e. bit + 1)
``state``   TTL line level after the edge (1 rising, 0 falling)
``word``    network event word (VStim event code)
``code``    DH5 event code: ``+line`` / ``-line`` for rising / falling TTL edges, ``word +``
            ``network_events_offset`` for network events (the codes of ``ephyscon convert``)
``name``    role of the TTL line (``[aux_channels]``) or VStim event name
==========  ===============================================================================

Lines whose role is in ``[events] exclude_roles`` (default the sync and wheel lines, which
switch thousands of times per minute) are left out, with ``lines_without_role = false`` also
all lines without a role (e.g. unconnected inputs that toggle).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ephyscon.sync import AUX

if TYPE_CHECKING:
    from ephyscon.session import Run
    from ephyscon.session_config import AuxChannelMap, EventsConfig
    from ephyscon.sync import RunSync

logger = logging.getLogger(__name__)

COLUMNS = ["time", "source", "line", "state", "word", "code", "name"]


def vstim_event_names() -> dict[int, str]:
    """VStim network event word -> name."""
    from vstim.network_event_codes import VStimEventCode

    return {int(code): name for name, code in VStimEventCode.asdict().items()}


def event_code_names(offset: int) -> dict[str, int]:
    """Names of the DH5 event codes of network events (``word + offset``)."""
    return {name: word + offset for word, name in vstim_event_names().items()}


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float if c == "time" else object) for c in COLUMNS})


def run_events(run: Run, sync: RunSync, aux_channels: AuxChannelMap, config: EventsConfig) -> pd.DataFrame:
    """TTL and network events of a run (see the module docs)."""
    if run.aux_stream is None:
        return empty_events()
    roles = {bit + 1: role for role, bit in aux_channels.digital.items()}
    excluded = {line for line, role in roles.items() if role in config.exclude_roles}
    wanted = None if config.lines_without_role else set(roles) - excluded
    parts = []
    try:
        ttl = run.format.read_ttl_events(run, wanted)
    except NotImplementedError:
        ttl = pd.DataFrame({"aux_sample": [], "line": [], "state": []})
    ttl = ttl[~ttl.line.isin(excluded)]
    if len(ttl):
        line, state = ttl.line.to_numpy(int), ttl.state.to_numpy(int)
        parts.append(pd.DataFrame({
            "time": sync.to_reference_time(AUX, ttl.aux_sample.to_numpy(float)), "source": "ttl",
            "line": line, "state": state, "word": pd.NA, "code": np.where(state > 0, line, -line),
            "name": [roles.get(n, "") for n in line],
        }))
    network = run.format.read_network_events(run)
    if len(network):
        names = vstim_event_names()
        word = network.word.to_numpy(int)
        parts.append(pd.DataFrame({
            "time": sync.to_reference_time(AUX, network.aux_sample.to_numpy(float)), "source": "network",
            "line": pd.NA, "state": pd.NA, "word": word, "code": word + config.network_events_offset,
            "name": [names.get(int(w), "") for w in word],
        }))
    if not parts:
        return empty_events()
    events = pd.concat(parts, ignore_index=True).sort_values("time", kind="stable", ignore_index=True)
    for column in ("line", "state", "word", "code"):
        events[column] = events[column].astype("Int64")
    logger.info("Run %s: %d TTL edges, %d network events (left out lines %s)", run.name,
                (events.source == "ttl").sum(), (events.source == "network").sum(), sorted(excluded) or "none")
    return events[COLUMNS]


def load_run_events(path) -> pd.DataFrame:
    events = pd.read_csv(path, keep_default_na=False, na_values={c: [""] for c in ("line", "state", "word", "code")})
    for column in ("line", "state", "word", "code"):
        events[column] = events[column].astype("Int64")
    return events
