"""Named pulses and markers of the trigger lines (``[events.lines]``, ``[[events.markers]]`` in
``session.toml``), from the events of the runs (``events/<run>.csv``).

* pulse: from the active edge of a line (rising, or falling for ``active = "low"``) to the
  next opposite edge, named after the line's role (DH5 ``/Intervals/<role>``, NWB
  ``intervals/trigger_pulses``, BIDS ``events.tsv`` with ``trial_type = <role>``). Two active
  edges in a row, or an active edge without an end in the run, give a pulse without end (left
  out of the DH5 intervals, duration ``n/a`` in BIDS); an inactive edge without a start (the
  line state at the start of the run) is dropped. As ``triggers.pulses`` of the Fritz/Mini
  BIDS conversions.
* marker: one time per pulse of a role (its onset or end) plus ``offset_ms``, or per trial
  (``role = "trial"``: its start or end); DH5 ``/Markers/<name>``, NWB ``markers`` table. A
  marker with an offset is derived (e.g. a stimulus onset at a fixed time in the trial), not
  measured; its description should say so. Every file with trials also gets ``trial_start``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ephyscon.session import Run
    from ephyscon.session_config import EventsConfig

TRIAL_ROLE = "trial"
TRIAL_START_MARKER = "trial_start"
PULSE_COLUMNS = ["start_time", "stop_time", "role", "line", "run"]


def active_state(role: str, config: EventsConfig) -> int:
    """Line level (0 or 1) while a pulse of ``role`` lasts."""
    line = config.lines.get(role)
    return 0 if line is not None and line.active == "low" else 1


def line_pulses(events: pd.DataFrame, config: EventsConfig, run: str = "") -> pd.DataFrame:
    """Pulses of the TTL lines with a role in the events of one run (columns ``PULSE_COLUMNS``,
    ``stop_time`` NaN for a pulse without end), sorted by start."""
    ttl = events[(events.source == "ttl") & (events["name"].fillna("") != "")]
    rows = []
    for (line, role), edges in ttl.groupby(["line", "name"], sort=True):
        active = active_state(str(role), config)
        start = None
        for time, state in zip(edges.time.to_numpy(float), edges.state.to_numpy(int)):
            if state == active:
                if start is not None:
                    rows.append((start, np.nan, role, int(line)))
                start = time
            elif start is not None:
                rows.append((start, time, role, int(line)))
                start = None
        if start is not None:
            rows.append((start, np.nan, role, int(line)))
    pulses = pd.DataFrame(rows, columns=PULSE_COLUMNS[:4])
    pulses["run"] = run
    pulses["line"] = pulses["line"].astype(np.int64)
    return pulses.sort_values(["start_time", "role"], kind="stable", ignore_index=True)


def markers(pulses: pd.DataFrame, trials: pd.DataFrame, config: EventsConfig) -> dict[str, tuple[np.ndarray, str]]:
    """Marker name -> (sorted times, description), from the pulses and trials of a file (same
    time axis)."""
    out: dict[str, tuple[np.ndarray, str]] = {}
    if not trials.empty:
        out[TRIAL_START_MARKER] = (np.sort(trials.start_time.to_numpy(float)), "start of the trials (TRIALMAP StartTime)")
    for marker in config.markers:
        if marker.role == TRIAL_ROLE:
            times = trials.start_time if marker.edge == "onset" else trials.stop_time
            times = times.to_numpy(float) if not trials.empty else np.zeros(0)
        else:
            selected = pulses[pulses.role == marker.role]
            times = (selected.start_time if marker.edge == "onset" else selected.stop_time).to_numpy(float)
        times = np.sort(times[np.isfinite(times)] + marker.offset_ms / 1000)
        edge = "start" if marker.edge == "onset" else "end"
        source = f"{edge} of the {'trials' if marker.role == TRIAL_ROLE else f'{marker.role} pulses'}"
        if marker.offset_ms:
            source += f" + {marker.offset_ms:g} ms (derived, not measured)"
        out[marker.name] = (times, f"{marker.description}; {source}" if marker.description else source)
    return out


def event_code_names(lines: dict[int, str], config: EventsConfig) -> list[tuple[int, str]]:
    """(DH5 event code, name) of the edges of the lines (``line -> role``): ``<role>_on`` for
    the active edge, ``<role>_off`` for the other (codes ``+line`` rising, ``-line`` falling)."""
    out = []
    for line, role in lines.items():
        on = line if active_state(role, config) else -line
        out += [(on, f"{role}_on"), (-on, f"{role}_off")]
    return sorted(out)


def line_descriptions(lines: dict[int, str], config: EventsConfig) -> dict[str, str]:
    """Role -> description of its pulses (with the line and its active level)."""
    out = {}
    for line, role in sorted(lines.items()):
        entry = config.lines.get(role)
        level = "low" if active_state(role, config) == 0 else "high"
        text = entry.description if entry is not None and entry.description else role
        out[role] = f"{text} (TTL line {line}, active {level})"
    return out


def session_pulses(work_dir: Path, runs: list[Run], offsets: dict[str, float],
                   config: EventsConfig) -> tuple[pd.DataFrame, dict[int, str]]:
    """Pulses of all runs on the file's time axis (``offsets``: start of each run on it) and
    the lines with a role (``line -> role``), from ``events/<run>.csv``."""
    from ephyscon.events import load_run_events

    parts, lines = [], {}
    for run in runs:
        path = work_dir / "events" / f"{run.name}.csv"
        if not path.exists():
            continue
        events = load_run_events(path)
        ttl = events[(events.source == "ttl") & (events["name"] != "")]
        lines.update({int(line): str(role) for line, role in ttl.groupby("line")["name"].first().items()})
        pulses = line_pulses(events, config, run.name)
        pulses[["start_time", "stop_time"]] += offsets[run.name]
        parts.append(pulses)
    pulses = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PULSE_COLUMNS)
    return pulses.sort_values(["start_time", "role"], kind="stable", ignore_index=True), lines
