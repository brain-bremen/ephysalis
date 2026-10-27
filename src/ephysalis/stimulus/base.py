"""Interface for stimulus systems.

A stimulus system knows (1) which paradigms were presented in a run, (2) which condition
was shown in each trial and (3) how trial times are marked in the recording (photodiode,
TTL words, network events). Trial times are resolved against the synchronised streams of
the run (:class:`~ephysalis.sync.RunSync`), so they end up on the reference clock.

Implementations register under the ``ephysalis.stimulus_systems`` entry point group. To
support a new stimulus system, implement :class:`StimulusSystem` and register it in the
``pyproject.toml`` of ephysalis or of any other installed package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from ephysalis.plugins import STIMULUS_SYSTEM_GROUP, available, get_plugin
from ephysalis.session import Run

if TYPE_CHECKING:
    from ephysalis.auxsignals import AuxSignals
    from ephysalis.runstart import RunStart
    from ephysalis.sync import RunSync


@dataclass
class Paradigm:
    name: str  # e.g. "RF" for 1_stimulusStageRF
    run_name: str
    index_in_run: int  # 0-based; runs can contain several paradigms
    system: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StimulusSystem(ABC):
    name: ClassVar[str]

    @abstractmethod
    def can_handle(self, run: Run) -> bool:
        """Whether this system finds stimulus information for ``run``."""

    @abstractmethod
    def discover(self, run: Run) -> list[Paradigm]:
        """List the paradigms presented in ``run``, in presentation order."""

    @abstractmethod
    def load_conditions(self, paradigm: Paradigm) -> pd.DataFrame:
        """One row per presented trial (in presentation order) with its stimulus parameters.

        Required columns: ``trial`` (0-based) and ``condition`` (system-specific condition id).
        All other columns are stimulus parameters and end up as columns of the NWB trials table.
        """

    def load_trials(
        self, run: Run, paradigm: Paradigm, sync: RunSync, aux: AuxSignals | None
    ) -> pd.DataFrame:
        """:meth:`load_conditions` plus ``start_time`` and ``stop_time`` in s on the reference
        clock of ``sync``. Trials without a detected time are dropped (with a warning)."""
        raise NotImplementedError(f"Stimulus system {self.name} does not provide trial times yet")

    def run_start(self, run: Run, trials: dict[str, pd.DataFrame]) -> RunStart | None:
        """A better start time of the run than the recording's, from the stimulus system's own
        clock and the run's trials (:meth:`load_trials` of its paradigms), or None
        (:mod:`ephysalis.runstart`)."""
        return None


def get_stimulus_system(name: str) -> StimulusSystem:
    return get_plugin(STIMULUS_SYSTEM_GROUP, name)


def detect_stimulus_system(run: Run) -> StimulusSystem | None:
    for name in available(STIMULUS_SYSTEM_GROUP):
        system = get_stimulus_system(name)
        if system.can_handle(run):
            return system
    return None
