"""The input formats that come with ephysalis, as typed entry point targets: a type checker
verifies each module against the Protocols of its capabilities (:mod:`ephysalis.plugins`).
Lab-specific formats and the metadata plugins are separate packages (e.g. ephysalis-bremen,
ephysalis-freiburg)."""

from typing import Protocol

from ephysalis.formats.input import openephys, spikeglx
from ephysalis.plugins import (
    HasSynchronisedTimestamps,
    InputFormat,
    LocatesSyncPulseChannels,
    ReadsAuxiliaryAnalogInputs,
    ReadsDigitalLineEdges,
    ReadsEventWords,
    ReadsTextMessages,
)


class SpikeGLXInputFormat(InputFormat, LocatesSyncPulseChannels, ReadsAuxiliaryAnalogInputs, ReadsDigitalLineEdges,
                          Protocol): ...


class OpenEphysInputFormat(InputFormat, HasSynchronisedTimestamps, ReadsAuxiliaryAnalogInputs, ReadsDigitalLineEdges,
                           ReadsTextMessages, ReadsEventWords, Protocol): ...


SPIKEGLX: SpikeGLXInputFormat = spikeglx
OPENEPHYS: OpenEphysInputFormat = openephys
