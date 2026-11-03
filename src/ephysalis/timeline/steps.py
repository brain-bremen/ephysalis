"""The steps that put a run's streams on one clock and extract what happened when.

``ephysalis align-clocks``
    clock of every stream on the run's reference clock (:func:`align_run_clocks`), from the
    sync-pulse edges (cached in ``<work>/sync``); report ``<work>/clocks/<run>.json``
``ephysalis extract-trials``
    trial tables of all paradigms of the stimulus system and the run start time from its clock
    (:func:`extract_run_trials`; ``trials/``, ``run_start/``)
``ephysalis extract-events``
    edges of the digital lines and network events (:mod:`ephysalis.timeline.events`; ``events/``)
``ephysalis extract-behavior``
    eye position and running speed at 1 kHz (:func:`extract_run_behavior`; ``eye/``,
    ``running_speed/``)

The three extract steps use the clock alignment (from the cached edges, extracted on demand
when ``align-clocks`` has not run) and are independent of each other. Everything is expressed
on the run's reference clock (the first probe unless configured otherwise) in seconds from the
start of that probe's recording.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ephysalis.timeline.auxsignals import AuxSignals, bin_means_on_reference, wheel_speed
from ephysalis.session.model import Run, StreamKind
from ephysalis.timeline.events import run_events
from ephysalis.stimulus.runstart import RunStart
from ephysalis.session.config import AuxChannelMap, EventsConfig, SessionConfig
from ephysalis.stimulus.base import Paradigm, detect_stimulus_system, get_stimulus_system
from ephysalis.timeline.clocks import AUX, RunSync, SyncEdgeCache, synchronize_run

if TYPE_CHECKING:
    from ephysalis.stimulus.base import StimulusSystem

logger = logging.getLogger(__name__)

RUNNING_SPEED_RATE = 1000.0  # Hz
# processing files of a session (sync edge cache, ...) go to <session>/ephysalis/
WORK_DIR_NAME = "ephysalis"
OLD_WORK_DIR_NAME = "ephyscon"  # before the rename to ephysalis


def default_work_dir(session_path: str | Path) -> Path:
    """``<session>/ephysalis``, or ``<session>/ephyscon`` of an existing work dir made before the rename."""
    new, old = Path(session_path) / WORK_DIR_NAME, Path(session_path) / OLD_WORK_DIR_NAME
    return old if old.is_dir() and not new.exists() else new


def default_sync_cache(session_path: str | Path) -> SyncEdgeCache:
    return SyncEdgeCache(default_work_dir(session_path) / "sync")


@dataclass
class RunTiming:
    run_name: str
    sync: RunSync
    paradigms: list[Paradigm] = field(default_factory=list)
    trials: dict[str, pd.DataFrame] = field(default_factory=dict)  # paradigm name -> trials
    running_speed: np.ndarray | None = None  # cm/s at RUNNING_SPEED_RATE, sample k = (k, k+1] ms
    eye_position: np.ndarray | None = None  # (samples, 2) x/y in V at RUNNING_SPEED_RATE, same bins
    events: pd.DataFrame | None = None  # see ephysalis.timeline.events
    run_start: RunStart | None = None  # better start time from the stimulus system (ephysalis.stimulus.runstart)

    def trial_samples(self, paradigm: str, stream: str, column: str = "start_time") -> np.ndarray:
        """Trial times of a paradigm as fractional sample indices of another stream."""
        return self.sync.to_samples(stream, self.trials[paradigm][column].to_numpy())


def aux_channel_map(run: Run, config: SessionConfig | None = None) -> AuxChannelMap:
    if config is not None:
        return config.aux_channels
    return run.format.default_aux_channels(run)


def align_run_clocks(run: Run, config: SessionConfig | None = None,
                     sync_cache: SyncEdgeCache | None = None) -> RunSync:
    """Clock of every stream of a run on the reference clock. Sync edges are taken from (and
    added to) ``sync_cache`` if given; without a cache they are extracted from the data every time."""
    return synchronize_run(run, aux_channel_map(run, config), cache=sync_cache)


def clock_report(sync: RunSync) -> dict[str, Any]:
    """The clock alignment of a run as JSON: per stream the linear fit (rate in samples per
    reference second, offset of sample 0 in s), the number of matched sync edges and the largest
    deviation of the edges from the fit (between edges, the map interpolates them exactly)."""
    streams = {}
    for stream, clock in sync.maps.items():
        slope, intercept = clock._fit
        streams[stream] = {
            "nominal_rate_hz": clock.sampling_frequency, "rate_hz": 1 / slope, "offset_s": intercept,
            "drift_ppm": (clock.sampling_frequency * slope - 1) * 1e6, "sync_edges": int(clock.source_samples.size),
            "max_residual_ms": clock.max_residual_s() * 1e3,
        }
    return {"reference": sync.reference, "streams": streams}


def write_clock_report(work_dir: Path, run_name: str, sync: RunSync) -> Path:
    import json

    path = Path(work_dir) / "clocks" / f"{run_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clock_report(sync), indent=1), encoding="utf-8")
    return path


def run_stimulus_system(run: Run, config: SessionConfig | None = None) -> StimulusSystem | None:
    run_config = config.run_config(run.name) if config is not None else None
    return (
        get_stimulus_system(run_config.stimulus_system)
        if run_config is not None and run_config.stimulus_system
        else detect_stimulus_system(run)
    )


def _aux(run: Run, config: SessionConfig | None) -> AuxSignals | None:
    return AuxSignals(run, aux_channel_map(run, config)) if run.aux_stream is not None else None


def extract_run_trials(run: Run, sync: RunSync, config: SessionConfig | None = None
                       ) -> tuple[list[Paradigm], dict[str, pd.DataFrame], RunStart | None]:
    """``(paradigms, trials per paradigm, run start)`` from the run's stimulus system."""
    system = run_stimulus_system(run, config)
    if system is None:
        logger.info("Run %s: no stimulus system found", run.name)
        return [], {}, None
    aux = _aux(run, config)
    paradigms = system.discover(run)
    trials = {paradigm.name: system.load_trials(run, paradigm, sync, aux) for paradigm in paradigms}
    return paradigms, trials, system.run_start(run, trials)


def extract_run_events(run: Run, sync: RunSync, config: SessionConfig | None = None) -> pd.DataFrame:
    """Edges of the digital lines and network events (:func:`ephysalis.timeline.events.run_events`)."""
    return run_events(run, sync, aux_channel_map(run, config), config.events if config is not None else EventsConfig())


def extract_run_behavior(run: Run, sync: RunSync, config: SessionConfig | None = None
                         ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(running speed, eye position)`` at RUNNING_SPEED_RATE on the reference clock, None
    where the run has no wheel / eye channels ([aux_channels])."""
    aux = _aux(run, config)
    if aux is None:
        return None, None
    reference = (run.get_recording(StreamKind.AUX) if sync.reference == AUX
                 else run.get_recording(StreamKind.AP, sync.reference))
    duration_s = reference.get_num_samples() / reference.sampling_frequency
    running_speed = eye_position = None
    if aux.has("wheel"):
        running_speed = wheel_speed(sync.to_reference_time(AUX, aux.wheel_rising_edges()), duration_s)
    if "eye_x" in aux.channels.analog and "eye_y" in aux.channels.analog:
        eye = np.stack([aux.analog_volts("eye_x"), aux.analog_volts("eye_y")], axis=1)
        bin_edges = np.arange(int(np.floor(duration_s * RUNNING_SPEED_RATE)) + 1) / RUNNING_SPEED_RATE
        eye_position = bin_means_on_reference(eye, sync.to_samples(AUX, bin_edges)).astype(np.float32)
    return running_speed, eye_position


def extract_run_timing(
    run: Run, config: SessionConfig | None = None, sync_cache: SyncEdgeCache | None = None
) -> RunTiming:
    """Clock alignment, trials, events and behaviour signals of a run in one go (``ephysalis
    inspect --timing``)."""
    sync = align_run_clocks(run, config, sync_cache)
    timing = RunTiming(run_name=run.name, sync=sync)
    timing.running_speed, timing.eye_position = extract_run_behavior(run, sync, config)
    timing.events = extract_run_events(run, sync, config)
    timing.paradigms, timing.trials, timing.run_start = extract_run_trials(run, sync, config)
    return timing
