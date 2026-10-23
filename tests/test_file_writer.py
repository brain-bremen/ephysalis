"""Unit tests for file_writer abstraction layer."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from dhspec.cont import create_channel_info
from dhspec.trialmap import TRIALMAP_DATASET_DTYPE

from ephyscon.config import DH5OutputOptions
from ephyscon.file_writer import (
    DH5Writer,
    FileWriter,
    create_file_writer,
)


class TestDH5Writer:
    """Unit tests for DH5Writer implementation."""

    def test_dh5_writer_initialization(self):
        """Test that DH5Writer initializes correctly."""
        with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = create_file_writer(
                filename=tmp_path,
                output_format="dh5",
                board_names=["TestBoard"],
                dh5_options=DH5OutputOptions(validate_structure=False),
            )

            assert isinstance(writer, DH5Writer)
            assert isinstance(writer, FileWriter)
            assert writer.filename.endswith(".dh5")

            writer.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_write_continuous_data(self):
        """Test writing continuous data to DH5 CONT block."""
        with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = create_file_writer(
                filename=tmp_path,
                output_format="dh5",
                board_names=["TestBoard"],
                dh5_options=DH5OutputOptions(validate_structure=False),
            )

            # Create test data
            n_samples = 1000
            data = np.random.randn(n_samples, 1).astype(np.int16)

            # Create channel info
            channel_info = create_channel_info(
                GlobalChanNumber=0,
                BoardChanNo=0,
                ADCBitWidth=16,
                MaxVoltageRange=10.0,
                MinVoltageRange=-10.0,
                AmplifChan0=0,
            )

            # Write continuous data
            writer.write_continuous_data(
                data=data,
                channel_info=channel_info,
                sample_rate_hz=30000.0,
                start_time_ns=0,
                channel_name="TestChannel",
                group_id=1,
                calibration=np.array([0.195]),
            )

            # Verify the CONT block was created
            h5file = writer._file
            assert "CONT1" in h5file  # CONT blocks are named CONT<id> at root level
            assert h5file["CONT1/DATA"].shape[0] == n_samples

            writer.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_write_event_triggers(self):
        """Test writing event triggers to DH5."""
        with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = create_file_writer(
                filename=tmp_path,
                output_format="dh5",
                board_names=["TestBoard"],
                dh5_options=DH5OutputOptions(validate_structure=False),
            )

            # Create test event data
            timestamps_ns = np.array([100000, 200000, 300000], dtype=np.int64)
            event_codes = np.array([1, 2, 3], dtype=np.int32)
            event_code_names = {"START": 1, "STOP": 2, "TRIGGER": 3}

            # Write event triggers
            writer.write_event_triggers(
                timestamps_ns=timestamps_ns,
                event_codes=event_codes,
                event_code_names=event_code_names,
            )

            # Verify events were written
            h5file = writer._file
            assert "/EV02" in h5file
            assert len(h5file["/EV02"]) == 3

            # Verify event code names were added as attributes
            for name, code in event_code_names.items():
                assert name in h5file["/EV02"].attrs
                assert h5file["/EV02"].attrs[name] == code

            writer.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_write_trialmap(self):
        """Test writing trial map to DH5."""
        with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = create_file_writer(
                filename=tmp_path,
                output_format="dh5",
                board_names=["TestBoard"],
                dh5_options=DH5OutputOptions(validate_structure=False),
            )

            n_trials = 5
            trial_data = np.recarray(shape=(n_trials,), dtype=TRIALMAP_DATASET_DTYPE)
            for i in range(n_trials):
                trial_data[i]["TrialNo"] = i + 1
                trial_data[i]["StartTime"] = np.int64(i * 1e9)
                trial_data[i]["EndTime"] = np.int64((i + 1) * 1e9)
                trial_data[i]["Outcome"] = 1

            writer.write_trialmap(trial_data=trial_data, tool_version="test_tool v1.0")

            h5file = writer._file
            assert "/TRIALMAP" in h5file
            assert len(h5file["/TRIALMAP"]) == n_trials

            ops = list(h5file["/Operations"].keys())
            assert any("Write trialmap" in k for k in ops)

            writer.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_add_operation(self):
        """Test adding operation to DH5 file."""
        with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = create_file_writer(
                filename=tmp_path,
                output_format="dh5",
                board_names=["TestBoard"],
                dh5_options=DH5OutputOptions(validate_structure=False),
            )

            # Add operation (note: file creation already adds 000_create_file operation)
            writer.add_operation(
                operation_name="Test Operation",
                tool_version="test_tool (v1.0.0)",
            )

            # Verify operation was added
            h5file = writer._file
            assert "/Operations" in h5file
            operation_groups = list(h5file["/Operations"].keys())
            assert len(operation_groups) == 2  # create_file + our operation
            assert any("Test Operation" in op_name for op_name in operation_groups)

            writer.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_add_operation_with_metadata(self):
        """Test adding operation with metadata attributes."""
        with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = create_file_writer(
                filename=tmp_path,
                output_format="dh5",
                board_names=["TestBoard"],
                dh5_options=DH5OutputOptions(validate_structure=False),
            )

            # Add operation with metadata
            metadata = {
                "param1": 42,
                "param2": "test_value",
            }

            writer.add_operation(
                operation_name="Test Operation With Metadata",
                tool_version="test_tool (v1.0.0)",
                metadata=metadata,
            )

            # Verify metadata was added as attributes
            h5file = writer._file
            operation_groups = list(h5file["/Operations"].keys())
            latest_operation = sorted(operation_groups)[-1]
            operation_group = h5file[f"/Operations/{latest_operation}"]

            for key, value in metadata.items():
                assert key in operation_group.attrs
                assert operation_group.attrs[key] == value

            writer.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_close(self):
        """Test that close properly closes the file."""
        with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = create_file_writer(
                filename=tmp_path,
                output_format="dh5",
                board_names=["TestBoard"],
                dh5_options=DH5OutputOptions(validate_structure=False),
            )

            # Close the writer
            writer.close()

            # Verify file is closed by trying to open it again
            with h5py.File(tmp_path, "r") as f:
                # Should be able to open successfully
                # Check for basic DH5 structure (Operations group is created on file creation)
                assert "/Operations" in f

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCreateFileWriter:
    """Test the factory function."""

    def test_create_dh5_writer(self):
        """Test creating a DH5 writer via factory."""
        with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = create_file_writer(
                filename=tmp_path,
                output_format="dh5",
                board_names=["TestBoard"],
            )

            assert isinstance(writer, DH5Writer)
            assert isinstance(writer, FileWriter)

            writer.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_create_dh5_writer_with_options(self):
        """Test creating DH5 writer with custom options."""
        with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            options = DH5OutputOptions(
                validate_structure=False,
            )

            writer = create_file_writer(
                filename=tmp_path,
                output_format="dh5",
                board_names=["TestBoard"],
                dh5_options=options,
            )

            assert writer._options.validate_structure is False

            writer.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_create_nwb_writer_not_implemented(self):
        """Test that NWB writer raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            create_file_writer(
                filename="test.nwb",
                output_format="nwb",
                board_names=["TestBoard"],
            )

    def test_unsupported_format(self):
        """Test that unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            create_file_writer(
                filename="test.dat",
                output_format="unsupported",
                board_names=["TestBoard"],
            )
