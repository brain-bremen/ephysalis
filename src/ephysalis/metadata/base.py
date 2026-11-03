"""Interface for metadata plugins: what is known about a session beyond the recording itself,
from the files of a lab setup (stimulus software, the lab's sheets and notes).

A plugin can provide

- **trials**: which paradigms were presented in a run (:meth:`MetadataPlugin.discover`), which
  condition was shown in each trial (:meth:`~MetadataPlugin.load_conditions`) and how trial times
  are marked in the recording (photodiode, TTL words, network events;
  :meth:`~MetadataPlugin.load_trials`), resolved against the synchronised streams of the run
  (:class:`~ephysalis.timeline.clocks.RunSync`) onto the reference clock (``extract-trials``);
- **run start times** from the stimulus software's clock (:meth:`~MetadataPlugin.run_start`);
- **config suggestions** for ``session.toml`` (:meth:`~MetadataPlugin.suggest_config`): probe
  areas and depths, bad channels, which runs to include, notes. They are applied once, by
  ``ephysalis init``; afterwards ``session.toml`` is the only source the steps read.

Implementations register under the ``ephysalis.metadata_plugins`` entry point group: implement
:class:`MetadataPlugin` and register it in the ``pyproject.toml`` of ephysalis or of any other
installed package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from ephysalis.plugins import METADATA_PLUGIN_GROUP, available, get_plugin
from ephysalis.session.model import Run

if TYPE_CHECKING:
    from ephysalis.metadata.runstart import RunStart
    from ephysalis.session.config import SessionConfig
    from ephysalis.session.model import Session
    from ephysalis.timeline.auxsignals import AuxSignals
    from ephysalis.timeline.clocks import RunSync


@dataclass
class Paradigm:
    name: str  # e.g. "RF" for 1_stimulusStageRF
    run_name: str
    index_in_run: int  # 0-based; runs can contain several paradigms
    system: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MetadataPlugin(ABC):
    name: ClassVar[str]

    @abstractmethod
    def can_handle(self, run: Run) -> bool:
        """Whether this plugin finds trial information for ``run``."""

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
        raise NotImplementedError(f"Metadata plugin {self.name} does not provide trial times yet")

    def run_start(self, run: Run, trials: dict[str, pd.DataFrame]) -> RunStart | None:
        """A better start time of the run than the recording's, from the stimulus software's own
        clock and the run's trials (:meth:`load_trials` of its paradigms), or None
        (:mod:`ephysalis.metadata.runstart`)."""
        return None

    def suggest_config(self, session: Session, config: SessionConfig, source: Path | None = None) -> list[str]:
        """Add what this plugin knows about ``session`` to ``config`` (``ephysalis init``: the
        defaults from the recording), from the setup's files in the session folder and, if given,
        ``source`` (e.g. a lab sheet outside the session folder). Returns warnings (e.g. values
        that disagree with the recording). Default: nothing to add."""
        return []


def get_metadata_plugin(name: str) -> MetadataPlugin:
    return get_plugin(METADATA_PLUGIN_GROUP, name)


def all_metadata_plugins() -> list[MetadataPlugin]:
    return [get_metadata_plugin(name) for name in available(METADATA_PLUGIN_GROUP)]


def detect_metadata_plugin(run: Run) -> MetadataPlugin | None:
    """The plugin that finds trial information for ``run`` (first in name order), or None."""
    return next((plugin for plugin in all_metadata_plugins() if plugin.can_handle(run)), None)
