"""The plugins that come with ephysalis, as typed entry point targets: a type checker verifies
each module against the Protocols of its capabilities (:mod:`ephysalis.plugins`)."""

from typing import Protocol

from ephysalis.formats.input import openephys, spikeglx
from ephysalis.metadata import vstim
from ephysalis.metadata.psychtoolbox_neuropixels_freiburg import plugin as psychtoolbox_neuropixels_freiburg
from ephysalis.plugins import (
    AddsSessionConfigFromMetadataFiles,
    ExtractsTrialTables,
    HasSynchronisedTimestamps,
    InputFormat,
    LocatesSyncPulseChannels,
    MetadataPlugin,
    NamesEventWords,
    ReadsAuxiliaryAnalogInputs,
    ReadsDigitalLineEdges,
    ReadsEventWords,
    ReadsTextMessages,
)


class SpikeGLXInputFormat(InputFormat, LocatesSyncPulseChannels, ReadsAuxiliaryAnalogInputs, ReadsDigitalLineEdges,
                          Protocol): ...


class OpenEphysInputFormat(InputFormat, HasSynchronisedTimestamps, ReadsAuxiliaryAnalogInputs, ReadsDigitalLineEdges,
                           ReadsTextMessages, ReadsEventWords, Protocol): ...


class VStimMetadataPlugin(MetadataPlugin, ExtractsTrialTables, NamesEventWords, Protocol): ...


class PsychtoolboxNeuropixelsFreiburgMetadataPlugin(MetadataPlugin, ExtractsTrialTables, AddsSessionConfigFromMetadataFiles, Protocol): ...


SPIKEGLX: SpikeGLXInputFormat = spikeglx
OPENEPHYS: OpenEphysInputFormat = openephys
VSTIM: VStimMetadataPlugin = vstim
PSYCHTOOLBOX_NEUROPIXELS_FREIBURG: PsychtoolboxNeuropixelsFreiburgMetadataPlugin = psychtoolbox_neuropixels_freiburg
