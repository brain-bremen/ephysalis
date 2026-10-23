"""Tests for trialmap.py module, especially Operation writing."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import h5py
import numpy as np
import pytest
from dh5io import DH5File
from dh5io.create import create_dh_file
from vstim.tdr import TrialOutcome

from ephyscon.events import Messages
from ephyscon.trialmap import (
    TrialEndMessage,
    TrialMapConfig,
    TrialStartMessage,
    parse_trial_end_message,
    parse_trial_start_message,
    process_oe_trialmap,
)


@pytest.fixture
def mock_recording():
    """Create a mock Recording object."""
    recording = Mock()
    recording.directory = Path("/fake/path")
    recording.info = {
        "events": [
            {
                "source_processor": "Message Center",
                "folder_name": "events",
                "stream_name": "messages",
            }
        ]
    }
    return recording


@pytest.fixture
def mock_messages():
    """Create mock trial messages."""
    from ephyscon.events import EventMetadata

    metadata = EventMetadata(
        channel_name="messages",
        folder_name="MessageCenter",
        identifier="message-center.messages",
        sample_rate=30000.0,
        stream_name="messages",
        type="Messages",
        description="Message Center messages",
        source_processor="Message Center",
    )

    messages = Messages(
        metadata=metadata,
        text=np.array(
            [
                b"VSTIM: TRIAL_START 1 TRIALTYPE 0 TIMESEQUENCE 0 FRAME 1032",
                b"VSTIM: TRIAL_END 1 TRIALTYPE 0 TIMESEQUENCE 0 FRAME 2048 OUTCOME 1",
                b"VSTIM: TRIAL_START 2 TRIALTYPE 1 TIMESEQUENCE 1 FRAME 3064",
                b"VSTIM: TRIAL_END 2 TRIALTYPE 1 TIMESEQUENCE 1 FRAME 4080 OUTCOME 5",
            ]
        ),
        timestamps=np.array([1.0, 2.0, 3.0, 4.0]),
        sample_numbers=np.array([30000, 60000, 90000, 120000]),
    )
    return messages


@pytest.fixture
def tmp_dh5_file(tmp_path):
    """Create a temporary DH5 file for testing."""
    dh5_path = tmp_path / "test.dh5"

    # Create a proper DH5 file with valid structure
    dh5file = create_dh_file(str(dh5_path), overwrite=True, validate=True)
    yield dh5file
    # DH5File will be closed automatically when the h5py file is closed


class TestTrialMessageParsing:
    """Tests for trial message parsing functions."""

    def test_parse_trial_start_message(self):
        msg = "TRIAL_START 1 TRIALTYPE 0 TIMESEQUENCE 0 FRAME 1032"
        result = parse_trial_start_message(msg)

        assert isinstance(result, TrialStartMessage)
        assert result.trial_index == 1
        assert result.trial_type_number == 0
        assert result.time_sequence_index == 0
        assert result.frame_number == 1032

    def test_parse_trial_end_message(self):
        msg = "TRIAL_END 1 TRIALTYPE 0 TIMESEQUENCE 0 FRAME 2048 OUTCOME 1"
        result = parse_trial_end_message(msg)

        assert isinstance(result, TrialEndMessage)
        assert result.trial_index == 1
        assert result.trial_type_number == 0
        assert result.frame_number == 2048
        assert result.outcome == TrialOutcome.Hit


class TestTrialmapOperation:
    """Tests for Operation creation in trialmap processing."""

    @patch("ephyscon.trialmap.get_messages_from_recording")
    def test_trialmap_creates_operation(
        self, mock_get_messages, tmp_dh5_file, mock_recording, mock_messages
    ):
        """Verify that process_oe_trialmap creates an Operation in the DH5 file."""
        from ephyscon.file_writer import DH5Writer

        mock_get_messages.return_value = mock_messages

        config = TrialMapConfig(
            use_message_center_messages=True,
            trial_start_ttl_line=None,
        )

        # Wrap DH5File in a DH5Writer
        file_writer = DH5Writer(tmp_dh5_file)

        # Process trialmap
        result_config = process_oe_trialmap(config, mock_recording, file_writer)

        # Check that Operation was created
        operations = list(tmp_dh5_file._file["/Operations"].keys())
        trialmap_ops = [op for op in operations if "Write trialmap" in op]

        assert len(trialmap_ops) == 1, (
            "Should create exactly one 'Write trialmap' operation"
        )

        # Check operation attributes
        operation_group = tmp_dh5_file._file[f"/Operations/{trialmap_ops[0]}"]
        assert "Date" in operation_group.attrs
        assert "Operator name" in operation_group.attrs
        assert "Tool" in operation_group.attrs
        tool_attr = operation_group.attrs["Tool"]
        if isinstance(tool_attr, bytes):
            tool_attr = tool_attr.decode()
        assert "ephyscon.trialmap" in tool_attr
        assert "(v" in tool_attr  # Check version format

