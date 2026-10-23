"""Per-run timing: clock sync, trial times of all paradigms and running speed.

Everything is expressed on the run's reference clock (the first probe unless configured
otherwise) in seconds from the start of that probe's recording.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ephyscon.auxsignals import AuxSignals, wheel_speed
from ephyscon.session import Run, StreamKind
from ephyscon.session_config import AuxChannelMap, SessionConfig
from ephyscon.stimulus.base import Paradigm, detect_stimulus_system, get_stimulus_system
from ephyscon.sync import AUX, RunSync, synchronize_run

logger = logging.getLogger(__name__)

RUNNING_SPEED_RATE = 1000.0  # Hz


@dataclass
class RunTiming:
    run_name: str
    sync: RunSync
    paradigms: list[Paradigm] = field(default_factory=list)
    trials: dict[str, pd.DataFrame] = field(default_factory=dict)  # paradigm name -> trials
    running_speed: np.ndarray | None = None  # cm/s at RUNNING_SPEED_RATE, sample k = (k, k+1] ms

    def trial_samples(self, paradigm: str, stream: str, column: str = "start_time") -> np.ndarray:
        """Trial times of a paradigm as fractional sample indices of another stream."""
        return self.sync.to_samples(stream, self.trials[paradigm][column].to_numpy())


def aux_channel_map(run: Run, config: SessionConfig | None = None) -> AuxChannelMap:
    if config is not None:
        return config.aux_channels
    return AuxChannelMap.legacy_default(run.extra.get("aux", {}).get("n_analog", 8))


def extract_run_timing(run: Run, config: SessionConfig | None = None) -> RunTiming:
    aux_channels = aux_channel_map(run, config)
    sync = synchronize_run(run, aux_channels)
    timing = RunTiming(run_name=run.name, sync=sync)

    aux = AuxSignals(run, aux_channels) if run.aux_stream is not None else None
    reference = run.get_recording(StreamKind.AP, sync.reference)
    duration_s = reference.get_num_samples() / reference.sampling_frequency

    if aux is not None and aux.has("wheel"):
        edges = sync.to_reference_time(AUX, aux.wheel_rising_edges())
        timing.running_speed = wheel_speed(edges, duration_s)

    run_config = config.run_config(run.name) if config is not None else None
    system = (
        get_stimulus_system(run_config.stimulus_system)
        if run_config is not None and run_config.stimulus_system
        else detect_stimulus_system(run)
    )
    if system is None:
        logger.info("Run %s: no stimulus system found", run.name)
        return timing
    timing.paradigms = system.discover(run)
    for paradigm in timing.paradigms:
        timing.trials[paradigm.name] = system.load_trials(run, paradigm, sync, aux)
    return timing
