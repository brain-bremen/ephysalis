import pytest
import numpy as np
import tempfile
import os
import h5py
from unittest.mock import Mock, patch

from ephyscon.decimation import decimate_raw_data, decimate_np_array, DecimationConfig
from open_ephys.analysis.recording import Recording, Continuous, ContinuousMetadata
from dh5io.create import create_dh_file
from dh5io.cont import validate_cont_group


def create_sinusoid_signal(
    n_samples,
    n_channels,
    sample_rate=30000,
    frequencies=(10, 50),
    amplitudes=(1.0, 0.5),
    noise_std=0.1,
    seed=42,
):
    """Create test signal with two sinusoids and noise.

    Args:
        n_samples: Number of samples
        n_channels: Number of channels
        sample_rate: Sampling rate in Hz
        frequencies: Tuple of two frequencies for the sinusoids (Hz)
        amplitudes: Tuple of amplitudes for the two sinusoids
        noise_std: Standard deviation of additive Gaussian noise
        seed: Random seed for reproducible noise


    """
    np.random.seed(seed)

    # Create time vector
    t = np.arange(n_samples) / sample_rate

    # Create the two sinusoids
    sin1 = amplitudes[0] * np.sin(2 * np.pi * frequencies[0] * t)
    sin2 = amplitudes[1] * np.sin(2 * np.pi * frequencies[1] * t)

    # Combine sinusoids
    signal = sin1 + sin2

    # Add noise
    noise = noise_std * np.random.randn(n_samples)
    signal_with_noise = signal + noise

    # Replicate across channels
    test_samples = np.column_stack([signal_with_noise] * n_channels)

    return test_samples, t


class MockContinuous(Continuous):
    def __init__(self, samples, metadata: ContinuousMetadata):
        self.samples = samples
        self.metadata = metadata
        self.timestamps = samples / metadata.sample_rate

    def get_samples(
        self,
        start_sample_index=0,
        end_sample_index=-1,
        selected_channels=None,
        selected_channel_names=None,
    ):
        if selected_channel_names and self.metadata.channel_names:
            # Find the index of the selected channel
            channel_idx = self.metadata.channel_names.index(selected_channel_names[0])
            # Return samples for that channel (reshape to column vector)
            return self.samples[:, channel_idx : channel_idx + 1]
        return self.samples


class MockRecording(Recording):
    def __init__(self, continuous_data_list):
        # Don't call super().__init__() to avoid complex initialization
        self._continuous = continuous_data_list
        self._events = None
        self._spikes = None

    @property
    def continuous(self):
        return self._continuous

    @continuous.setter
    def continuous(self, value):
        self._continuous = value

    @property
    def events(self):
        return self._events

    @property
    def spikes(self):
        return self._spikes

    # Implement abstract methods from Recording class
    def load_spikes(self, experiment_id=0, recording_id=0):
        pass

    def load_events(self, experiment_id=0, recording_id=0):
        pass

    def load_continuous(self, experiment_id=0, recording_id=0):
        pass

    def load_messages(self, experiment_id=0, recording_id=0):
        pass

    @staticmethod
    def detect_format(directory):
        return True

    def detect_recordings(self, mmap_timestamps=True):
        pass

    def read_sync_channel(self, experiment_id=0, recording_id=0):
        pass

    def read_stream_sync_channel(self, stream_name, experiment_id=0, recording_id=0):
        pass

    def __str__(self):
        return None

    def _get_experiments(self):
        return []

    def _get_recordings(self, experiment_id):
        return []

    def _get_processors(self, experiment_id, recording_id):
        return []

    def _get_streams(self, experiment_id, recording_id, processor_id):
        return []


class TestDecimateNpArray:
    """Tests for the decimate_np_array function"""

    def test_decimate_np_array_basic(self):
        """Test basic decimation functionality"""
        # Create test data with sinusoids and noise
        data, t = create_sinusoid_signal(
            n_samples=1000,
            n_channels=2,
            frequencies=(10, 40),
            amplitudes=(1.0, 0.5),
            noise_std=0.1,
            seed=42,
        )

        # Test decimation
        result = decimate_np_array(
            data=data,
            downsampling_factor=10,
            filter_order=30,
            filter_type="fir",
            axis=0,
            zero_phase=True,
        )

        # Check that output is decimated
        assert result.shape[0] == data.shape[0] // 10
        assert result.shape[1] == data.shape[1]

    def test_decimate_np_array_different_factors(self):
        """Test decimation with different downsampling factors"""
        data, t = create_sinusoid_signal(
            n_samples=1000,
            n_channels=1,
            frequencies=(15, 35),
            amplitudes=(1.2, 0.3),
            noise_std=0.05,
            seed=42,
        )

        for factor in [2, 5, 10, 20]:
            result = decimate_np_array(
                data=data,
                downsampling_factor=factor,
                filter_order=10,
                filter_type="fir",
                axis=0,
                zero_phase=True,
            )
            expected_length = data.shape[0] // factor
            assert result.shape[0] == expected_length

    def test_decimate_np_array_filter_types(self):
        """Test decimation with different filter types"""
        data, t = create_sinusoid_signal(
            n_samples=500,
            n_channels=1,
            frequencies=(12, 48),
            amplitudes=(0.8, 0.4),
            noise_std=0.08,
            seed=42,
        )

        for ftype in ["fir", "iir"]:
            result = decimate_np_array(
                data=data,
                downsampling_factor=5,
                filter_order=20,
                filter_type=ftype,
                axis=0,
                zero_phase=True,
            )
            assert result.shape[0] == data.shape[0] // 5


@pytest.mark.skip(reason="Not ready yet")
def test_decimation_config_with_real_dh5file(plt):
    """Test DecimationConfig integration with a real temporary DH5File"""
    # Create a temporary file for the DH5 file
    with tempfile.NamedTemporaryFile(suffix=".dh5", delete=False) as temp_file:
        temp_path = temp_file.name

    # Create a real DH5File using create_dh_file
    dh5file = create_dh_file(temp_path, overwrite=True, validate=True)

    # Create test data with two sinusoids and noise
    test_samples, t = create_sinusoid_signal(
        n_samples=30000,
        n_channels=2,
        sample_rate=30000,
        frequencies=(10, 200),  # 10 Hz and 50 Hz sinusoids
        amplitudes=(6.0, 5.0),  # Different amplitudes
        noise_std=0,
        seed=42,
    )

    # Create metadata and continuous data
    metadata = ContinuousMetadata(
        channel_names=["CH1", "CH2"],
        sample_rate=30000,
        source_node_name="test_node",
        source_node_id=100,
        stream_name="test_stream",
        num_channels=2,
        bit_volts=[0.05, 0.05],
    )

    continuous = MockContinuous(samples=test_samples, metadata=metadata)
    recording = MockRecording([continuous])

    # Create config with custom values
    config = DecimationConfig(
        downsampling_factor=10,
        ftype="fir",
        filter_order=30,
        zero_phase=True,
        included_channel_names=None,
        start_block_id=2001,
        scale_max_abs_to=None,
    )

    # Test that the config works with the actual decimate_raw_data function
    result_config = decimate_raw_data(config, recording, dh5file)

    # Verify the configuration was updated correctly
    assert result_config.downsampling_factor == 10
    assert result_config.ftype == "fir"
    assert result_config.zero_phase
    assert result_config.filter_order == 30
    assert result_config.included_channel_names == ["CH1", "CH2"]
    assert result_config.start_block_id == 2001

    # Check that data was actually written to the DH5 file
    expected_decimated_length = test_samples.shape[0] // config.downsampling_factor

    assert 2001 in dh5file.get_cont_group_ids()
    assert 2002 in dh5file.get_cont_group_ids()

    dh5file.get_cont_group_by_id(2001)
    cont_ch1 = dh5file.get_cont_group_by_id(2001)

    data = cont_ch1.calibrated_data
    assert data is not None
    assert data.size == expected_decimated_length

    assert data.mean() < 0.1
    assert data.max() <= 6.0 + 5.0

    t_decimated = np.arange(0, expected_decimated_length) / (
        metadata.sample_rate / config.downsampling_factor
    )
    plt.plot(t_decimated, cont_ch1.calibrated_data, "o-")
    plt.plot(t, test_samples[:, 0], ".")

    del dh5file

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.mark.skip(reason="Not ready yet")
class TestDecimateRawDataIntegration:
    """Integration tests for decimate_raw_data function"""

    @patch("dhspec.cont.create_channel_info")
    @patch("dh5io.cont.create_cont_group_from_data_in_file")
    @patch("dh5io.operations.add_operation_to_file")
    @patch("ephyscon.version.get_version_from_pyproject")
    def test_decimate_raw_data_basic(
        self,
        mock_version,
        mock_index_array,
        mock_add_operation,
        mock_create_cont_group,
        mock_channel_info,
    ):
        """Test basic functionality of decimate_raw_data"""
        # Setup mocks
        mock_version.return_value = "1.0.0"
        mock_index_array.return_value = np.array([0])
        mock_channel_info.return_value = {"test": "channel_info"}

        # Create test data with two sinusoids and noise
        test_samples, t = create_sinusoid_signal(
            n_samples=1000,
            n_channels=2,
            frequencies=(20, 100),
            amplitudes=(1.5, 0.8),
            noise_std=0.3,
            seed=42,
        )

        # Create mock metadata and continuous data
        metadata = ContinuousMetadata(
            channel_names=["CH1", "CH2"],
            sample_rate=30000,
            source_node_name="test_node",
            source_node_id=100,
            stream_name="test_stream",
            num_channels=2,
            bit_volts=[0.05, 0.05],
        )

        continuous = MockContinuous(samples=test_samples, metadata=metadata)
        recording = MockRecording([continuous])

        # Create mock DH5File
        mock_dh5file = Mock()
        mock_dh5file._file = Mock()

        # Create config
        config = DecimationConfig(
            downsampling_factor=10,
            ftype="fir",
            filter_order=30,
            zero_phase=True,
            start_block_id=2001,
        )

        # Call the function
        result_config = decimate_raw_data(config, recording, mock_dh5file)

        # Verify results
        assert result_config.included_channel_names == ["CH1", "CH2"]

        # Verify that the DH5 operations were called for each channel
        assert mock_create_cont_group.call_count == 2  # One for each channel
        assert mock_add_operation.call_count == 1

        # Verify channel info creation
        assert mock_channel_info.call_count == 2

    @patch("dhspec.cont.create_channel_info")
    @patch("dh5io.cont.create_cont_group_from_data_in_file")
    @patch("dh5io.operations.add_operation_to_file")
    @patch("ephyscon.version.get_version_from_pyproject")
    def test_decimate_raw_data_channel_selection(
        self,
        mock_version,
        mock_index_array,
        mock_add_operation,
        mock_create_cont_group,
        mock_channel_info,
    ):
        """Test decimate_raw_data with specific channel selection"""
        # Setup mocks
        mock_version.return_value = "1.0.0"
        mock_index_array.return_value = np.array([0])
        mock_channel_info.return_value = {"test": "channel_info"}

        # Create test data with two sinusoids and noise
        test_samples, t = create_sinusoid_signal(
            n_samples=500,
            n_channels=3,
            frequencies=(15, 60),
            amplitudes=(1.0, 0.6),
            noise_std=0.2,
            seed=42,
        )

        # Create mock metadata and continuous data
        metadata = ContinuousMetadata(
            channel_names=["CH1", "CH2", "CH3"],
            sample_rate=30000,
            source_node_name="test_node",
            source_node_id=100,
            stream_name="test_stream",
            num_channels=3,
            bit_volts=[0.05, 0.05, 0.05],
        )

        continuous = MockContinuous(samples=test_samples, metadata=metadata)
        recording = MockRecording([continuous])

        # Create mock DH5File
        mock_dh5file = Mock()
        mock_dh5file._file = Mock()

        # Create config with specific channel selection
        config = DecimationConfig(
            downsampling_factor=5,
            included_channel_names=["CH1", "CH3"],  # Only process 2 out of 3 channels
            start_block_id=2001,
        )

        # Call the function
        result_config = decimate_raw_data(config, recording, mock_dh5file)

        # Verify results
        assert result_config.included_channel_names == ["CH1", "CH3"]

        # Verify that the DH5 operations were called only for selected channels
        assert mock_create_cont_group.call_count == 2  # Only CH1 and CH3
        assert mock_add_operation.call_count == 1

    def test_decimate_raw_data_no_continuous_data(self):
        """Test decimate_raw_data raises assertion error when no continuous data"""
        # Create recording with no continuous data
        recording = MockRecording([])
        recording.continuous = None

        # Create mock DH5File
        mock_dh5file = Mock()

        # Create config
        config = DecimationConfig()

        # Should raise assertion error
        with pytest.raises(AssertionError, match="No continuous data found"):
            decimate_raw_data(config, recording, mock_dh5file)

    @patch("dhspec.cont.create_channel_info")
    @patch("dh5io.cont.create_cont_group_from_data_in_file")
    @patch("dh5io.operations.add_operation_to_file")
    @patch("ephyscon.version.get_version_from_pyproject")
    def test_decimate_raw_data_multiple_continuous_streams(
        self,
        mock_version,
        mock_index_array,
        mock_add_operation,
        mock_create_cont_group,
        mock_channel_info,
    ):
        """Test decimate_raw_data with multiple continuous streams"""
        # Setup mocks
        mock_version.return_value = "1.0.0"
        mock_index_array.return_value = np.array([0])
        mock_channel_info.return_value = {"test": "channel_info"}

        # Create test data for two streams with sinusoids and noise
        test_samples1, t = create_sinusoid_signal(
            n_samples=300,
            n_channels=2,
            frequencies=(25, 75),
            amplitudes=(1.2, 0.7),
            noise_std=0.25,
            seed=42,
        )
        test_samples2, t = create_sinusoid_signal(
            n_samples=300,
            n_channels=1,
            frequencies=(30, 80),
            amplitudes=(0.9, 0.4),
            noise_std=0.15,
            seed=43,
        )

        # Create mock metadata and continuous data for stream 1
        metadata1 = ContinuousMetadata(
            channel_names=["A1", "A2"],
            sample_rate=30000,
            source_node_name="node1",
            source_node_id=101,
            stream_name="stream1",
            num_channels=2,
            bit_volts=[0.05, 0.05],
        )
        continuous1 = MockContinuous(samples=test_samples1, metadata=metadata1)

        # Create mock metadata and continuous data for stream 2
        metadata2 = ContinuousMetadata(
            channel_names=["B1"],
            sample_rate=30000,
            source_node_name="node2",
            source_node_id=102,
            stream_name="stream2",
            num_channels=1,
            bit_volts=[0.1],
        )
        continuous2 = MockContinuous(samples=test_samples2, metadata=metadata2)

        # Create recording with multiple streams
        recording = MockRecording([continuous1, continuous2])

        # Create mock DH5File
        mock_dh5file = Mock()
        mock_dh5file._file = Mock()

        # Create config
        config = DecimationConfig(downsampling_factor=3)

        # Call the function
        result_config = decimate_raw_data(config, recording, mock_dh5file)

        # Verify results
        assert result_config.included_channel_names == ["A1", "A2", "B1"]

        # Verify that the DH5 operations were called for all channels across all streams
        assert mock_create_cont_group.call_count == 3  # A1, A2, B1
        assert mock_add_operation.call_count == 1
