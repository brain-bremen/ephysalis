import logging
import os
import pprint
import warnings
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pathlib import Path

import numpy as np
from open_ephys.analysis.formats.BinaryRecording import BinaryRecording
from open_ephys.analysis.recording import Recording
from vstim.network_event_codes import VStimEventCode

import ephyscon.version
from ephyscon.file_writer import FileWriter

logger = logging.getLogger(__name__)


class EventPreprocessingConfig(BaseModel):
    network_events_offset: int = Field(
        default=1000,
        title="Network events offset",
        description="Value added to all VStim network event codes to avoid collisions with TTL line codes",
    )
    network_events_code_name_map: dict[str, int] | None = Field(
        default_factory=lambda: VStimEventCode.asdict(),
        title="Network event code map",
        description="Mapping from VStim event name to integer code. Leave empty to use VStim defaults",
    )
    ttl_line_names: dict[str, int] | None = Field(
        default=None,
        title="TTL line names",
        description="Custom mapping from TTL line name to event code. Leave empty to use line indices as codes",
    )


@dataclass
class EventMetadata:
    channel_name: str
    folder_name: str
    identifier: str
    sample_rate: float
    stream_name: str  # asdljkalskdjalskjdalskjd asdlkjasd asdlajskdas
    type: str
    description: str
    source_processor: str
    initial_state: int = 0


@dataclass
class Messages:
    metadata: EventMetadata
    text: np.ndarray  # of str
    sample_numbers: np.ndarray
    timestamps: np.ndarray

    def __iter__(self):
        # merge three nd arrays (text, sample_numbers and timestamps) to iterate over
        for i in range(len(self.text)):
            yield {
                "text": self.text[i].decode(),
                "sample_number": self.sample_numbers[i],
                "timestamp": self.timestamps[i],
            }

    def __str__(self):
        metadata_str = pprint.pformat(self.metadata)
        return (
            f"Messages Data: {self.metadata.folder_name}')\n"
            f"├─── Text: {self.text.shape}\n"
            f"├─── Sample Numbers: {self.sample_numbers.shape}\n"
            f"└─── {metadata_str}\n"
        )

    @staticmethod
    def from_folder(full_event_folder_path: str | Path, metadata: EventMetadata):
        return Messages(
            metadata=metadata,
            text=np.load(os.path.join(full_event_folder_path, "text.npy")),
            sample_numbers=np.load(
                os.path.join(full_event_folder_path, "sample_numbers.npy")
            ),
            timestamps=np.load(os.path.join(full_event_folder_path, "timestamps.npy")),
        )


@dataclass
class Event:
    metadata: EventMetadata
    full_words: np.ndarray
    timestamps: np.ndarray
    states: np.ndarray
    sample_numbers: np.ndarray

    def __len__(self):
        assert (
            len(self.full_words)
            == len(self.timestamps)
            == len(self.states)
            == len(self.sample_numbers)
        ), (
            f"Length mismatch: {len(self.full_words)}, {len(self.timestamps)}, {len(self.states)}, {len(self.sample_numbers)}"
        )
        return len(self.full_words)

    @staticmethod
    def from_folder(full_event_folder_path: str | Path, metadata: EventMetadata):
        return Event(
            metadata=metadata,
            full_words=np.load(os.path.join(full_event_folder_path, "full_words.npy")),
            timestamps=np.load(os.path.join(full_event_folder_path, "timestamps.npy")),
            states=np.load(os.path.join(full_event_folder_path, "states.npy")),
            sample_numbers=np.load(
                os.path.join(full_event_folder_path, "sample_numbers.npy")
            ),
        )

    def __init__(
        self, metadata: EventMetadata, full_words, timestamps, states, sample_numbers
    ):
        # verify all args have same length
        assert (
            len(full_words) == len(timestamps) == len(states) == len(sample_numbers)
        ), (
            f"Length mismatch: {len(full_words)}, {len(timestamps)}, {len(states)}, {len(sample_numbers)}"
        )
        self.metadata = metadata
        self.full_words = full_words
        self.timestamps = timestamps
        self.states = states
        self.sample_numbers = sample_numbers

    def __str__(self):
        metadata_str = pprint.pformat(self.metadata)
        return (
            f"Event Data: {self.metadata.folder_name}')\n"
            f"├─── Full Words: {self.full_words.shape}\n"
            f"├─── Timestamps: {self.timestamps.shape}\n"
            f"├─── States: {self.states.shape}\n"
            f"├─── Sample Numbers: {self.sample_numbers.shape}\n"
            f"└─── {metadata_str}\n"
        )


@dataclass
class FullWordEvent:
    metadata: EventMetadata
    full_words: np.ndarray
    timestamps: np.ndarray
    sample_numbers: np.ndarray

    def __len__(self):
        assert (
            len(self.full_words) == len(self.timestamps) == len(self.sample_numbers)
        ), (
            f"Length mismatch: {len(self.full_words)}, {len(self.timestamps)}, {len(self.sample_numbers)}"
        )
        return len(self.full_words)


def remove_repeating_simultaneous_words(event: Event) -> FullWordEvent:
    """Convert event data to contain only changing words.

    The Network Events plugin can send full words, which causes
    multiple lines to change in the same sample. This function
    removes the mulitple event entries for each bit and removes
    the state attribute.

    """

    # TODO: This does not work. Write a test that verifies it does.
    unique_indices = np.where(np.diff(event.full_words, prepend=np.nan) != 0)[0]

    return FullWordEvent(
        metadata=event.metadata,
        full_words=event.full_words[unique_indices],
        timestamps=event.timestamps[unique_indices],
        sample_numbers=event.sample_numbers[unique_indices],
    )


def event_from_eventfolder(
    recording_directory: str | Path, metadata: EventMetadata
) -> Event | Messages | FullWordEvent:
    # full_event_folder_path = os.path.join(path, "events", metadata.folder_name)
    full_event_folder_path = (
        Path(recording_directory) / "events" / Path(metadata.folder_name)
    )

    assert os.path.exists(full_event_folder_path), (
        f"Events folder {full_event_folder_path} does not exist"
    )

    # return data based on metadata.source_processor
    match metadata.source_processor:
        case "Network Events":
            return remove_repeating_simultaneous_words(
                Event.from_folder(full_event_folder_path, metadata)
            )
        case "Message Center":
            return Messages.from_folder(full_event_folder_path, metadata)
        case "NI-DAQmx":
            return Event.from_folder(full_event_folder_path, metadata)
        case _:
            warnings.warn(
                f"Unsupported source processor: {metadata.source_processor}. Attempting generic Event loading..."
            )
            return Event.from_folder(full_event_folder_path, metadata)


def find_ev02_source(oeinfo: dict):
    for event in oeinfo["events"]:
        if (
            (
                event["source_processor"] == "NI-DAQmx"
                and event["stream_name"] == "PXIe-6341"
            )
            or event["source_processor"] == "Acquisition Board"
            or event["identifier"] == "acq-board.rhythm.events"
        ):
            return EventMetadata(**event)
    return None


def find_marker_source(oeinfo: dict):
    """Network Events for Markers"""
    for event in oeinfo["events"]:
        if event["source_processor"] == "Network Events":
            return EventMetadata(**event)


def process_oe_events(
    event_config: EventPreprocessingConfig, recording: Recording, file_writer: FileWriter
):
    logger.info(f"Processing events in {file_writer.filename}")

    timestamps_ns = np.array([], dtype=np.int64)
    event_codes = np.array([], dtype=np.int32)

    assert isinstance(recording, BinaryRecording), (
        "Recording must be a BinaryRecording to process events."
    )

    # TTL
    ev02_source_metadata = find_ev02_source(recording.info)
    if ev02_source_metadata is not None:
        logging.info(
            f"Processing TTL triggers from {ev02_source_metadata.stream_name} stream"
        )
        network_events_words = event_from_eventfolder(
            recording_directory=recording.directory,
            metadata=ev02_source_metadata,
        )
        assert isinstance(network_events_words, Event)

        timestamps_ns = np.array(
            np.int64(np.round(network_events_words.timestamps * 1e9)), dtype=np.int64
        )
        event_codes = network_events_words.states

    # Network Events
    network_events_source = find_marker_source(recording.info)
    network_events_offset = event_config.network_events_offset
    if network_events_source is not None:
        logger.info(
            f"Processing Network Events from {network_events_source.stream_name} stream"
        )
        network_events_words = event_from_eventfolder(
            recording_directory=recording.directory,
            metadata=network_events_source,
        )
        assert isinstance(network_events_words, FullWordEvent)

        # append to timesatamps_ns and event_codes
        timestamps_ns = np.concatenate(
            (timestamps_ns, np.int64(np.round(network_events_words.timestamps * 1e9)))
        ).astype(np.int64)
        event_codes = np.concatenate(
            (event_codes, network_events_words.full_words + network_events_offset)
        ).astype(np.int32)

    # sort event_codes and timesamps_ns according to timestamps_ns
    sort_indices = np.argsort(timestamps_ns)
    timestamps_ns = timestamps_ns[sort_indices]
    event_codes = event_codes[sort_indices]

    assert all(np.diff(timestamps_ns) >= 0)

    # Collect event code names
    event_code_names: dict[str, int] = {}

    if event_config.ttl_line_names is not None:
        for event_name, event_code in event_config.ttl_line_names.items():
            event_code_names[str(event_name)] = event_code + network_events_offset

    if network_events_source is not None and event_config.network_events_code_name_map is not None:
        logging.debug(
            f"Adding network events code names with offset {network_events_offset}"
        )
        for event_name, event_code in event_config.network_events_code_name_map.items():
            event_code_names[str(event_name)] = event_code + network_events_offset

    # Write events using file writer abstraction
    file_writer.write_event_triggers(
        timestamps_ns=timestamps_ns,
        event_codes=event_codes,
        event_code_names=event_code_names if event_code_names else None,
    )

    # Add operation using file writer abstraction
    file_writer.add_operation(
        operation_name="Process events",
        tool_version=f"ephyscon.events (v{ephyscon.version.get_version_from_pyproject()})",
    )

    return event_config
