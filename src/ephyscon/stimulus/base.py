"""Interface for stimulus systems.

A stimulus system knows (1) which paradigms were presented in a run and (2) which condition
was shown in each trial. Trial *timing* (photodiode, TTL words, network events) is resolved
against the synchronised event streams in a later step, so the same stimulus system can be
used with different acquisition setups.

Implementations register under the ``ephyscon.stimulus_systems`` entry point group. To
support a new stimulus system, implement :class:`StimulusSystem` and register it in the
``pyproject.toml`` of ephyscon or of any other installed package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from ephyscon.plugins import STIMULUS_SYSTEM_GROUP, available, get_plugin
from ephyscon.session import Run


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


def get_stimulus_system(name: str) -> StimulusSystem:
    return get_plugin(STIMULUS_SYSTEM_GROUP, name)


def detect_stimulus_system(run: Run) -> StimulusSystem | None:
    for name in available(STIMULUS_SYSTEM_GROUP):
        system = get_stimulus_system(name)
        if system.can_handle(run):
            return system
    return None
