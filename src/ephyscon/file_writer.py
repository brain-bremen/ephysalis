"""Format-agnostic file writer abstraction for multi-format output support.

This module provides an abstraction layer that allows OEcon to write to different
output formats (DH5, NWB, etc.) using a common interface. Format selection happens
via dependency injection - the appropriate writer is created once and injected into
all processing modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class FileWriter(ABC):
    """Abstract interface for format-agnostic file writers.

    Pure abstract base class with no implementation, logic, or data.
    Concrete implementations (DH5Writer, NWBWriter) must implement all methods.
    """

    @abstractmethod
    def write_continuous_data(
        self,
        data: np.ndarray,
        channel_info: Any,
        sample_rate_hz: float,
        start_time_ns: int,
        channel_name: str,
        group_id: int,
        calibration: np.ndarray | float,
    ) -> None:
        """Write continuous recording data (RAW, LFP, or MUA).

        Args:
            data: Continuous data array, shape (n_samples, n_channels)
            channel_info: Channel metadata (format-specific structure)
            sample_rate_hz: Sampling rate in Hz
            start_time_ns: Start timestamp in nanoseconds
            channel_name: Name/label for this channel/group
            group_id: CONT block ID (DH5) or processing module selector (NWB)
            calibration: Calibration factor(s) for converting to physical units
        """
        pass

    @abstractmethod
    def write_event_triggers(
        self,
        timestamps_ns: np.ndarray,
        event_codes: np.ndarray,
        event_code_names: dict[str, int] | None = None,
    ) -> None:
        """Write event triggers with optional event code name mappings.

        Args:
            timestamps_ns: Event timestamps in nanoseconds
            event_codes: Event codes corresponding to timestamps
            event_code_names: Optional mapping of event names to codes
        """
        pass

    @abstractmethod
    def write_trialmap(self, trial_data: np.ndarray, tool_version: str) -> None:
        """Write trial map and the accompanying operation entry.

        Args:
            trial_data: Structured array with trial information
            tool_version: Tool/version string recorded in the operation entry
        """
        pass

    @abstractmethod
    def add_operation(
        self,
        operation_name: str,
        tool_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record processing operation in file history.

        Args:
            operation_name: Name of the processing operation
            tool_version: Tool/version string (e.g., "ephyscon.raw (v0.2.0)")
            metadata: Optional additional metadata to store
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close file and flush all writes."""
        pass

    @property
    @abstractmethod
    def filename(self) -> str:
        """Get the output filename."""
        pass


class DH5Writer(FileWriter):
    """DH5 format writer implementation.

    Wraps dh5io library calls to match the FileWriter interface.
    """

    def __init__(self, dh5file: Any, options: Any | None = None):
        """Initialize DH5 writer.

        Args:
            dh5file: DH5File object from dh5io.create.create_dh_file()
            options: DH5-specific output options (DH5OutputOptions from ephyscon.config)
        """
        from ephyscon.config import DH5OutputOptions

        self._dh5file = dh5file  # Keep reference to DH5File wrapper
        self._file = dh5file._file  # h5py.File for compatibility
        self._options = options if options is not None else DH5OutputOptions()

    @property
    def filename(self) -> str:
        """Get the output filename."""
        return self._file.filename

    def write_continuous_data(
        self,
        data: np.ndarray,
        channel_info: Any,
        sample_rate_hz: float,
        start_time_ns: int,
        channel_name: str,
        group_id: int,
        calibration: np.ndarray | float,
    ) -> None:
        """Write continuous data to DH5 CONT block."""
        import dh5io.cont
        from dhspec.cont import create_empty_index_array

        # Create index array with single segment
        index = create_empty_index_array(1)
        index[0]["time"] = np.int64(start_time_ns)
        index[0]["offset"] = 0

        # Ensure calibration is an array
        if not isinstance(calibration, np.ndarray):
            calibration = np.array(calibration)

        # Write to DH5 CONT block
        dh5io.cont.create_cont_group_from_data_in_file(
            file=self._file,
            cont_group_id=group_id,
            data=data,
            index=index,
            sample_period_ns=np.int32(1e9 / sample_rate_hz),
            name=channel_name,
            channels=channel_info,
            calibration=calibration,
        )

    def write_event_triggers(
        self,
        timestamps_ns: np.ndarray,
        event_codes: np.ndarray,
        event_code_names: dict[str, int] | None = None,
    ) -> None:
        """Write event triggers to DH5 /EV02 dataset."""
        import dh5io.event_triggers
        from dhspec.event_triggers import EV_DATASET_NAME

        # Write event triggers
        dh5io.event_triggers.add_event_triggers_to_file(
            self._file, timestamps_ns=timestamps_ns, event_codes=event_codes
        )

        # Add event code name mappings as attributes if provided
        if event_code_names:
            ev_dataset = self._file[EV_DATASET_NAME]
            for name, code in event_code_names.items():
                ev_dataset.attrs[name] = np.int32(code)

    def write_trialmap(self, trial_data: np.ndarray, tool_version: str) -> None:
        """Write trial map to DH5 /TRIALMAP dataset and record the operation."""
        import dh5io.trialmap

        dh5io.trialmap.add_trialmap_to_file(self._file, trial_data)
        dh5io.trialmap.add_write_trialmap_operation(self._file, tool=tool_version)

    def add_operation(
        self,
        operation_name: str,
        tool_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add operation to DH5 /Operations group."""
        import dh5io.operations

        # Add operation to file
        dh5io.operations.add_operation_to_file(
            file=self._file,
            new_operation_group_name=operation_name,
            tool=tool_version,
        )

        # Add metadata as attributes if provided
        if metadata:
            operation_groups = list(self._file["/Operations"].keys())
            latest_operation = sorted(operation_groups)[-1]
            operation_group = self._file[f"/Operations/{latest_operation}"]

            for key, value in metadata.items():
                operation_group.attrs[key] = value

    def close(self) -> None:
        """Close the HDF5 file."""
        self._file.close()


def create_file_writer(
    filename: str | Path,
    output_format: str,
    board_names: list[str],
    dh5_options: Any | None = None,
    nwb_options: Any | None = None,
) -> FileWriter:
    """Factory function to create format-specific writer.

    Returns a FileWriter instance that can be injected into processing modules.
    Format-specific options are used only by the corresponding writer.

    Args:
        filename: Output file path
        output_format: Output format ("dh5" or "nwb")
        board_names: List of board names for metadata
        dh5_options: DH5-specific options (DH5OutputOptions, used only if output_format is "dh5")
        nwb_options: NWB-specific options (NWBOutputOptions, used only if output_format is "nwb")

    Returns:
        FileWriter instance for the specified format

    Raises:
        ValueError: If output_format is not supported
    """
    match output_format.lower():
        case "dh5":
            import dh5io.create

            from ephyscon.config import DH5OutputOptions

            if dh5_options is None:
                dh5_options = DH5OutputOptions()

            # Create DH5 file
            dh5file = dh5io.create.create_dh_file(
                str(filename),
                overwrite=True,
                boards=board_names,
                validate=dh5_options.validate_structure,
            )

            return DH5Writer(dh5file, options=dh5_options)

        case "nwb":
            # NWB implementation will be added in Phase 3
            raise NotImplementedError("NWB output format is not yet implemented")

        case _:
            raise ValueError(f"Unsupported output format: {output_format}")
