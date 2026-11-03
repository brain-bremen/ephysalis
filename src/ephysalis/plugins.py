"""The plugin interface of ephysalis: input formats and metadata plugins.

A plugin is any object with the right functions, typically a module of plain functions,
registered under an entry point group in the ``pyproject.toml`` of ephysalis or of any other
installed package (e.g. a lab's own plugin repository)::

    [project.entry-points."ephysalis.input_formats"]
    daq = "ephysalis_bremen.registry:DAQ"

The interface is defined by :class:`typing.Protocol` classes: a small required core per plugin
kind (:class:`InputFormat`, :class:`MetadataPlugin`) and optional capabilities (e.g.
:class:`ReadsDigitalLineEdges`, :class:`ExtractsTrialTables`). A plugin provides a capability
by defining all of its functions. Type checkers (mypy, pyright) verify a plugin module against
the Protocols where it is assigned to a variable of a Protocol type, e.g. in the plugin
package's registry module::

    class DaqInputFormat(InputFormat, ReadsDigitalLineEdges, ComputesChannelGainCalibration, Protocol): ...
    DAQ: DaqInputFormat = daq

When a plugin is loaded, ephysalis checks its required core, the plugin API version and every
capability it partly provides (all functions present, same parameters). A plugin that fails is
skipped with a warning, so the other plugins keep working. ephysalis asks for a capability with
``isinstance(plugin, Capability)``.

The interface is alpha: :data:`PLUGIN_API_VERSION` changes with every incompatible change, and a
plugin states the version it was written for (``plugin_api_version``).
"""

from __future__ import annotations

import inspect
import logging
import typing
from collections.abc import Collection
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from spikeinterface.core import BaseRecording

    from ephysalis.metadata.runstart import RunStart
    from ephysalis.session.config import AuxChannelMap, SessionConfig
    from ephysalis.session.model import Run, Session, StreamKind
    from ephysalis.timeline.auxsignals import AuxSignals
    from ephysalis.timeline.clocks import RunSync, SyncSource

PLUGIN_API_VERSION = 2  # 2: input_format_name / metadata_plugin_name instead of name
INPUT_FORMAT_GROUP = "ephysalis.input_formats"
METADATA_PLUGIN_GROUP = "ephysalis.metadata_plugins"

logger = logging.getLogger(__name__)


# --- input formats -------------------------------------------------------------------------

@runtime_checkable
class InputFormat(Protocol):
    """Required core of an input format: open a session folder and hand out lazy recordings."""

    input_format_name: str  # the name it is registered under (entry point), e.g. "spikeglx"
    plugin_api_version: int

    def can_read_session_folder(self, session_path: Path) -> bool:
        """Cheap check whether ``session_path`` looks like a session of this format."""
        ...

    def open_session(self, session_path: Path) -> Session:
        """All runs, probes and streams of a session (metadata only, no samples read)."""
        ...

    def get_stream_recording(self, run: Run, kind: StreamKind, probe_id: str | None) -> BaseRecording:
        """A lazy spikeinterface recording of one stream of a run (a probe's AP or LF band, or
        the auxiliary stream)."""
        ...

    def default_auxiliary_channel_roles(self, run: Run | None) -> AuxChannelMap:
        """Roles of the auxiliary channels (sync, photodiode, wheel, ...) when ``session.toml``
        defines none."""
        ...


@runtime_checkable
class ListsStreamChannelIds(Protocol):
    def list_stream_channel_ids(self, run: Run, stream: str) -> list[str] | None:
        """Channel ids recorded in ``stream`` of ``run`` from the metadata (no samples read), or
        None if the format cannot tell cheaply; probes are left out of runs lacking channels."""
        ...


@runtime_checkable
class LocatesSyncPulseChannels(Protocol):
    def locate_sync_pulse_channel(self, run: Run, probe_id: str | None,
                                  auxiliary_channel_roles: AuxChannelMap | None) -> SyncSource:
        """Channel and bit carrying the sync pulses of a probe's stream, or of the auxiliary
        stream (``probe_id`` None); the streams are aligned by matching these pulses."""
        ...


@runtime_checkable
class HasSynchronisedTimestamps(Protocol):
    def clock_alignment_from_synchronised_timestamps(self, run: Run, reference_stream: str | None
                                                     ) -> RunSync | None:
        """Clock alignment of all streams from timestamps the acquisition system already
        synchronised (e.g. Open Ephys), or None if the run needs sync pulses."""
        ...


@runtime_checkable
class ReadsAuxiliaryAnalogInputs(Protocol):
    def auxiliary_sampling_frequency(self, run: Run) -> float:
        """Sampling rate of the auxiliary stream (Hz)."""
        ...

    def read_auxiliary_analog_channel(self, run: Run, index: int) -> np.ndarray:
        """Raw ADC counts of analog input ``index`` of the auxiliary stream (whole run)."""
        ...

    def auxiliary_analog_volts_per_count(self, run: Run, index: int) -> float:
        """Volts per ADC count of analog input ``index``."""
        ...

    def auxiliary_analog_recording(self, run: Run) -> BaseRecording:
        """Lazy recording of the analog inputs of the auxiliary stream (gains in µV per count),
        e.g. for the raw data in the outputs."""
        ...


@runtime_checkable
class ReadsDigitalLineEdges(Protocol):
    def read_auxiliary_digital_line(self, run: Run, bit: int) -> np.ndarray:
        """One digital line (bit of the digital word) of the auxiliary stream as a bool array."""
        ...

    def read_digital_line_edges(self, run: Run, lines: Collection[int] | None) -> pd.DataFrame:
        """Edges of the digital lines (``lines``: only these, 1-based; None: all):
        ``aux_sample``, ``line`` (1-based, bit + 1), ``state`` (1 rising, 0 falling), in time order."""
        ...


@runtime_checkable
class ReadsTextMessages(Protocol):
    def read_text_messages(self, run: Run) -> pd.DataFrame:
        """Text messages recorded with the run (e.g. Open Ephys Message Center, Neuralynx event
        strings; sent by the stimulus software): ``aux_sample`` and ``text``."""
        ...


@runtime_checkable
class ReadsEventWords(Protocol):
    def read_event_words(self, run: Run) -> pd.DataFrame:
        """Event codes recorded with the run besides the digital line edges (e.g. Open Ephys network
        events, strobed words): ``aux_sample`` and ``word``."""
        ...


@runtime_checkable
class ReadsTrialDescriptorEvents(Protocol):
    def read_trial_descriptor_events(self, run: Run) -> pd.DataFrame:
        """Trial and stimulus numbers the acquisition system recorded at each trial start (e.g.
        ``TD01`` records, MC_Rack trigger events): ``aux_sample``, ``trial_number``,
        ``stimulus_number``."""
        ...


@runtime_checkable
class ComputesChannelGainCalibration(Protocol):
    def compute_channel_gain_calibration(self, session: Session, config: SessionConfig, work_dir: Path
                                         ) -> dict[str, Any] | None:
        """Gain of every probe channel (µV per ADC count) from the session's calibration
        recording (``[calibration]``), written to the work dir; None if there is none."""
        ...

    def apply_channel_gain_calibration(self, session: Session, config: SessionConfig | None,
                                       config_dir: Path | None, work_dir: Path) -> None:
        """Put the computed gains on the probe channels of all runs (every step that uses gains)."""
        ...


INPUT_FORMAT_CAPABILITIES: tuple[type, ...] = (
    ListsStreamChannelIds, LocatesSyncPulseChannels, HasSynchronisedTimestamps, ReadsAuxiliaryAnalogInputs,
    ReadsDigitalLineEdges, ReadsTextMessages, ReadsEventWords, ReadsTrialDescriptorEvents,
    ComputesChannelGainCalibration,
)


# --- metadata plugins ----------------------------------------------------------------------

@runtime_checkable
class MetadataPlugin(Protocol):
    """Required core of a metadata plugin: what is known about a session beyond the recording,
    from the files of a lab setup (stimulus software, the lab's sheets). It provides at least one
    of the capabilities below."""

    metadata_plugin_name: str  # the name it is registered under (entry point), e.g. "vstim"
    plugin_api_version: int


@runtime_checkable
class ExtractsTrialTables(Protocol):
    def extract_trial_tables(self, run: Run, clock_alignment: RunSync | None,
                             auxiliary_signals: AuxSignals | None) -> dict[str, pd.DataFrame] | None:
        """One table per paradigm presented in the run, in presentation order: one row per trial
        (``trial``, ``condition``, optionally ``trial_number``, the stimulus software's own trial
        number, and the stimulus parameters) and, with ``clock_alignment``, ``start_time`` /
        ``stop_time`` in s on the reference clock. None: not a run of this setup."""
        ...


@runtime_checkable
class EstimatesRunStartTime(Protocol):
    def estimate_run_start_time(self, run: Run, trial_tables: dict[str, pd.DataFrame]) -> RunStart | None:
        """A more accurate start date and time of the run than the recording's, from the stimulus
        software's own clock and the run's trials, or None."""
        ...


@runtime_checkable
class AddsSessionConfigFromMetadataFiles(Protocol):
    def add_session_config_from_metadata_files(self, session: Session, config: SessionConfig,
                                               metadata_file: Path | None) -> list[str]:
        """Add what the setup's files say about the session (probe areas and depths, bad channels,
        which runs to include, notes) to ``config``, once (``ephysalis init``); returns warnings.
        ``metadata_file``: a file outside the session folder (``init --metadata``)."""
        ...


@runtime_checkable
class NamesEventWords(Protocol):
    def name_event_words(self, run: Run) -> dict[int, str] | None:
        """Names of the event words of the run (:class:`ReadsEventWords`, e.g. VStim's event codes), or None."""
        ...


METADATA_PLUGIN_CAPABILITIES: tuple[type, ...] = (
    ExtractsTrialTables, EstimatesRunStartTime, AddsSessionConfigFromMetadataFiles, NamesEventWords,
)

_GROUPS = {INPUT_FORMAT_GROUP: (InputFormat, INPUT_FORMAT_CAPABILITIES),
           METADATA_PLUGIN_GROUP: (MetadataPlugin, METADATA_PLUGIN_CAPABILITIES)}
_NAME_MEMBER = {InputFormat: "input_format_name", MetadataPlugin: "metadata_plugin_name"}


# --- loading and checking ------------------------------------------------------------------

def _members(protocol: type) -> list[str]:
    return sorted(typing.get_protocol_members(protocol)) if hasattr(typing, "get_protocol_members") \
        else sorted(protocol.__protocol_attrs__)  # type: ignore[attr-defined]


def _parameters(function: Any) -> list[str]:
    return [parameter_name for parameter_name, parameter in inspect.signature(function).parameters.items()
            if parameter_name != "self" and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]


def plugin_problems(plugin: Any, core: type, capabilities: tuple[type, ...]) -> list[str]:
    """What is wrong with ``plugin``: missing members of the required core, a different plugin API
    version, capabilities provided only partly, functions with other parameters than the Protocol."""
    problems = [f"missing {member_name}" for member_name in _members(core) if not hasattr(plugin, member_name)]
    version = getattr(plugin, "plugin_api_version", None)
    if version is not None and version != PLUGIN_API_VERSION:
        problems.append(f"written for plugin API version {version}, this is ephysalis's version {PLUGIN_API_VERSION}")
    provided = [capability for capability in capabilities
                if any(hasattr(plugin, member_name) for member_name in _members(capability))]
    for capability in provided:
        missing = [member_name for member_name in _members(capability) if not hasattr(plugin, member_name)]
        if missing:
            problems.append(f"{capability.__name__} only partly provided: missing {', '.join(missing)}")
    for protocol in (core, *provided):
        for member_name in _members(protocol):
            expected, got = getattr(protocol, member_name, None), getattr(plugin, member_name, None)
            if callable(expected) and callable(got) and _parameters(got) != _parameters(expected):
                problems.append(f"{member_name}({', '.join(_parameters(got))}) instead of "
                                f"{member_name}({', '.join(_parameters(expected))})")
    if core is MetadataPlugin and not provided:
        problems.append("provides none of the metadata capabilities")
    return problems


@cache
def _load_group(group: str) -> dict[str, Any]:
    core, capabilities = _GROUPS[group]
    plugins = {}
    for entry_point in entry_points(group=group):
        try:
            plugin = entry_point.load()
        except Exception as error:  # noqa: BLE001 (one broken plugin must not disable the others)
            logger.warning("plugin %s (%s) not loaded: %s", entry_point.name, entry_point.value, error)
            continue
        problems = plugin_problems(plugin, core, capabilities)
        declared_name = getattr(plugin, _NAME_MEMBER[core], None)
        if declared_name is not None and declared_name != entry_point.name:
            problems.append(f"{_NAME_MEMBER[core]} {declared_name!r} differs from its entry point name")
        if problems:
            logger.warning("plugin %s (%s) skipped: %s", entry_point.name, entry_point.value, "; ".join(problems))
            continue
        plugins[entry_point.name] = plugin
    return plugins


def available(group: str) -> list[str]:
    return sorted(_load_group(group))


def get_plugin(group: str, plugin_name: str) -> Any:
    plugins = _load_group(group)
    if plugin_name not in plugins:
        raise KeyError(f"Unknown plugin {plugin_name!r} in {group}. Available: {sorted(plugins)}")
    return plugins[plugin_name]


def plugin_name(plugin: object) -> str:
    """The name an input format or metadata plugin is registered under."""
    return str(getattr(plugin, "input_format_name", None) or getattr(plugin, "metadata_plugin_name", None) or plugin)


Capability = TypeVar("Capability")


def require(plugin: object, capability: type[Capability], purpose: str) -> Capability:
    """``plugin`` as a ``capability`` (Protocol), or a TypeError saying what is missing for ``purpose``."""
    if not isinstance(plugin, capability):
        raise TypeError(f"{plugin_name(plugin)!r} has no {capability.__name__} ({', '.join(_members(capability))}),"
                        f" needed for {purpose}")
    return plugin


def input_formats() -> list[InputFormat]:
    return [get_plugin(INPUT_FORMAT_GROUP, format_name) for format_name in available(INPUT_FORMAT_GROUP)]


def metadata_plugins() -> list[MetadataPlugin]:
    return [get_plugin(METADATA_PLUGIN_GROUP, metadata_plugin_name)
            for metadata_plugin_name in available(METADATA_PLUGIN_GROUP)]
