import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy.signal as signal
from dhspec.cont import create_channel_info
from open_ephys.analysis.recording import Recording as OERecording
from pydantic import BaseModel, Field, field_validator

import ephyscon.default_mappings as default
import ephyscon.version
from ephyscon.decimation import DecimationConfig, decimate_np_array
from ephyscon.file_writer import FileWriter

logger = logging.getLogger(__name__)


class FilterConfigBA(BaseModel):
    b: list[float] | None
    a: list[float] | None

    @field_validator("b", "a", mode="before")
    @classmethod
    def coerce_ndarray_to_list(cls, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v


class ContinuousMuaConfig(BaseModel):
    highpass_cutoff_hz: float = Field(
        default=300.0,
        title="High-pass cutoff (Hz)",
        description="Cutoff frequency for the Butterworth high-pass filter applied before rectification",
    )
    filter_coecfficients_b_a: FilterConfigBA | None = Field(
        default=None,
        title="Filter coefficients (b, a)",
        description="Pre-computed filter coefficients (b, a). Auto-computed from the cutoff frequency if left empty",
    )
    included_channel_names: list[str] | None = Field(
        default=None,
        title="Included channels",
        description="Channel names to process. Leave empty to include all channels",
    )
    start_block_id: int = Field(
        default=default.DEFAULT_CONT_GROUP_RANGES[default.ContGroups.ESA][0],
        title="Start CONT block ID",
        description="First DH5 CONT block ID for MUA output (default range: 4001–5000)",
    )


def extract_continuous_mua(
    config: ContinuousMuaConfig,
    decimation_config: DecimationConfig,
    recording: OERecording,
    file_writer: FileWriter,
    on_channel: "Callable[[int, int], None] | None" = None,
) -> ContinuousMuaConfig:
    assert recording.continuous is not None, (
        "No continuous data found in the recording."
    )

    global_channel_index = 0
    dh5_cont_id = config.start_block_id

    total_channels = sum(
        sum(1 for name in (c.metadata.channel_names or [])
            if config.included_channel_names is None or name in config.included_channel_names)
        for c in recording.continuous
    )
    ch_done = 0

    for oe_cont in recording.continuous:
        oe_metadata = oe_cont.metadata

        assert oe_metadata.channel_names is not None, (
            "Channel names are not set in OE data."
        )

        if config.included_channel_names is None:
            config.included_channel_names = oe_metadata.channel_names

        decimation_config.included_channel_names = config.included_channel_names

        logger.info(
            f"Extracting continuous MUA from {oe_metadata.num_channels} channels continuous data from {oe_metadata.source_node_name} (source_node={oe_metadata.source_node_id})"
        )
        for channel_index, channel_name in enumerate(oe_metadata.channel_names):
            if channel_name not in config.included_channel_names:
                continue

            samples = oe_cont.get_samples(
                start_sample_index=0,
                end_sample_index=-1,
                selected_channels=None,
                selected_channel_names=[channel_name],
            )

            # High-pass filter
            if config.filter_coecfficients_b_a is None:
                b, a = signal.butter(
                    N=4,
                    Wn=config.highpass_cutoff_hz,
                    btype="highpass",
                    fs=oe_metadata.sample_rate,
                )
                config.filter_coecfficients_b_a = FilterConfigBA(b=b, a=a)
            filtered = signal.filtfilt(
                b=np.array(config.filter_coecfficients_b_a.b),
                a=np.array(config.filter_coecfficients_b_a.a),
                x=samples,
                axis=0,
            )

            # Rectify
            rectified = np.abs(filtered)

            # Decimate
            decimated_samples = decimate_np_array(
                data=rectified,
                downsampling_factor=decimation_config.downsampling_factor,
                filter_order=decimation_config.filter_order,
                filter_type=decimation_config.ftype,
                axis=0,
                zero_phase=decimation_config.zero_phase,
            )

            channel_info = create_channel_info(
                GlobalChanNumber=global_channel_index,
                BoardChanNo=channel_index,
                ADCBitWidth=16,
                MaxVoltageRange=10.0,
                MinVoltageRange=10.0,
                AmplifChan0=0,
            )

            scaling_factor = oe_cont.metadata.bit_volts[channel_index]
            decimated_samples /= scaling_factor
            decimated_samples = decimated_samples.astype(np.int16)

            # Write MUA continuous data using file writer abstraction
            file_writer.write_continuous_data(
                data=decimated_samples,
                channel_info=channel_info,
                sample_rate_hz=oe_metadata.sample_rate / decimation_config.downsampling_factor,
                start_time_ns=np.int64(oe_cont.timestamps[0] * 1e9),
                channel_name=f"{oe_metadata.stream_name}/{channel_name}/MUA",
                group_id=dh5_cont_id,
                calibration=np.array(oe_metadata.bit_volts[channel_index]),
            )

            dh5_cont_id += 1
            global_channel_index += 1
            ch_done += 1
            if on_channel:
                on_channel(ch_done, total_channels)

    # Add operation using file writer abstraction
    file_writer.add_operation(
        operation_name="Extract continuous MUA",
        tool_version=f"ephyscon.mua (v{ephyscon.version.get_version_from_pyproject()})",
    )

    return config
