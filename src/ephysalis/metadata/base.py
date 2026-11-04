"""Metadata plugins in the pipeline: which plugin describes a run, its trial tables and run start.

The interface is in :mod:`ephysalis.plugins` (:class:`~ephysalis.plugins.MetadataPlugin` and the
capabilities :class:`~ephysalis.plugins.ExtractsTrialTables`,
:class:`~ephysalis.plugins.EstimatesRunStartTime`,
:class:`~ephysalis.plugins.AddsSessionConfigFromMetadataFiles`,
:class:`~ephysalis.plugins.NamesEventWords`). :class:`ExperimentalProtocol` is a helper for plugins that
first list a run's experimental protocols and then read each one's trials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ephysalis.plugins import (
    METADATA_PLUGIN_GROUP,
    EstimatesRunStartTime,
    ExtractsTrialTables,
    MetadataPlugin,
    get_plugin,
    metadata_plugins,
)

if TYPE_CHECKING:
    from ephysalis.metadata.runstart import RunStart
    from ephysalis.recording_session.config import RecordingSessionConfig
    from ephysalis.recording_session.model import Run
    from ephysalis.timeline.auxsignals import AuxSignals
    from ephysalis.timeline.clocks import RunSync


@dataclass
class ExperimentalProtocol:
    """One experimental protocol presented in a run (helper for plugins)."""

    experimental_protocol_name: str  # e.g. "RF" for 1_stimulusStageRF; the key of its trial table
    run_name: str
    index_in_run: int  # 0-based; runs can contain several experimental protocols
    metadata_plugin_name: str  # the plugin that found it
    metadata_path: Path | None = None  # the file or folder the experimental protocol is read from (e.g. the .tdr)
    experimental_protocol_metadata: dict[str, Any] = field(default_factory=dict)  # copied into the trial table's attrs


NO_METADATA_PLUGIN = "none"  # metadata_plugin = "none": the run has no metadata plugin


def get_metadata_plugin(metadata_plugin_name: str) -> MetadataPlugin:
    return get_plugin(METADATA_PLUGIN_GROUP, metadata_plugin_name)


def run_trial_tables(run: Run, clock_alignment: RunSync | None, auxiliary_signals: AuxSignals | None,
                     config: RecordingSessionConfig | None = None) -> tuple[MetadataPlugin | None, dict[str, pd.DataFrame]]:
    """``(plugin, trial tables)`` of a run: from the plugin set for the run in ``session.toml``
    (``runs[].metadata_plugin``, else the session's ``metadata_plugin``; ``"none"``: no plugin), else from the
    first plugin (in name order) that recognises the run; ``(None, {})`` if none does."""
    run_config = config.run_config(run.name) if config is not None else None
    name = (run_config.metadata_plugin if run_config is not None and run_config.metadata_plugin
            else config.metadata_plugin if config is not None else None)
    if name == NO_METADATA_PLUGIN:
        return None, {}
    candidates = [get_metadata_plugin(name)] if name else metadata_plugins()
    for plugin in candidates:
        if isinstance(plugin, ExtractsTrialTables):
            tables = plugin.extract_trial_tables(run, clock_alignment, auxiliary_signals)
            if tables is not None:
                return plugin, tables
    return None, {}


def run_start_time(plugin: MetadataPlugin | None, run: Run, trial_tables: dict[str, pd.DataFrame]) -> RunStart | None:
    """The run start time from ``plugin``'s clock (:class:`~ephysalis.plugins.EstimatesRunStartTime`), or None."""
    if isinstance(plugin, EstimatesRunStartTime):
        return plugin.estimate_run_start_time(run, trial_tables)
    return None
