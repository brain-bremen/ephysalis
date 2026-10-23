import logging
from collections.abc import Callable
from enum import StrEnum

import dhspec.cont
import numpy as np
import scipy.signal as signal
from open_ephys.analysis.recording import Recording
from pydantic import BaseModel, Field, field_validator

import ephyscon.version
from ephyscon.scaling import scale_to_16_bit_range
from ephyscon.file_writer import FileWriter

logger = logging.getLogger(__name__)


class FilterType(StrEnum):
    FIR = "fir"
    IIR = "iir"


class DecimationConfig(BaseModel):
    downsampling_factor: int = Field(
        default=30,
        title="Downsampling factor",
        description="Factor by which the sample rate is reduced (e.g. 30 → 1 kHz LFP from 30 kHz raw)",
    )
    ftype: FilterType = Field(
        default=FilterType.FIR,
        title="Filter type",
        description="Anti-alias filter type: FIR (linear phase, recommended) or IIR (faster but with phase distortion)",
    )
    zero_phase: bool = Field(
        default=True,
        title="Zero phase",
        description="Apply filter twice (forward + backward) to eliminate phase distortion",
    )
    filter_order: int | None = Field(
        default=600,
        title="Filter order",
        description="Number of taps (FIR) or poles (IIR). Higher = sharper cutoff but slower. Leave empty for scipy default",
    )
    included_channel_names: list[str] | None = Field(
        default=None,
        title="Included channels",
        description="Channel names to process. Leave empty to include all channels",
    )
    start_block_id: int = Field(
        default=2001,
        title="Start CONT block ID",
        description="First DH5 CONT block ID for LFP output (default range: 2001–3000)",
    )
    scale_max_abs_to: int | None = Field(
        default=None,
        title="Scale max abs to",
        description="If set, rescale so the maximum absolute value maps to this integer. Leave empty to use original bit_volts scaling",
    )

    @field_validator("downsampling_factor")
    @classmethod
    def factor_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("downsampling_factor must be >= 1")
        return v


def decimate_np_array(
    data, downsampling_factor, filter_order, filter_type, axis, zero_phase: bool
):
    return signal.decimate(
        x=data,
        q=downsampling_factor,
        n=filter_order,
        ftype=filter_type,
        axis=axis,
        zero_phase=zero_phase,
    )


def decimate_raw_data(
    config: DecimationConfig,
    recording: Recording,
    file_writer: FileWriter,
    on_channel: "Callable[[int, int], None] | None" = None,
) -> DecimationConfig:
    assert recording.continuous is not None, (
        "No continuous data found in the recording."
    )

    global_channel_index = 0
    dh5_cont_id = config.start_block_id
    included_channel_names: list[str] = []

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
            logger.debug("No channel selection provided, selecting all channels")
            included_channel_names.extend(oe_metadata.channel_names)
        else:
            included_channel_names.extend(config.included_channel_names)

        # TODO: Use chunks of channels in parallel
        logger.info(
            f"Decimating ({oe_metadata.sample_rate} -> {oe_metadata.sample_rate / config.downsampling_factor} Hz) {oe_metadata.num_channels} channels continuous data from {oe_metadata.source_node_name} ({oe_metadata.source_node_id})"
        )
        for channel_index, channel_name in enumerate(oe_metadata.channel_names):
            # skip channel if not in included channels
            if channel_name not in included_channel_names:
                continue

            samples = oe_cont.get_samples(
                start_sample_index=0,
                end_sample_index=-1,
                selected_channels=None,
                selected_channel_names=[channel_name],
            )
            # samples x channels
            decimated_samples = decimate_np_array(
                data=samples,
                downsampling_factor=config.downsampling_factor,
                filter_order=config.filter_order,
                filter_type=config.ftype,
                axis=0,
                zero_phase=config.zero_phase,
            )

            channel_info = dhspec.cont.create_channel_info(
                GlobalChanNumber=global_channel_index,
                BoardChanNo=channel_index,
                ADCBitWidth=16,
                MaxVoltageRange=10.0,
                MinVoltageRange=10.0,
                AmplifChan0=0,
            )

            logger.debug(f"Data range: {np.min(samples)} - {np.max(samples)}")
            if config.scale_max_abs_to is not None:
                decimated_samples, scaling_factor = scale_to_16_bit_range(
                    decimated_samples
                )
            else:
                # use original scaling factor (bit_volts)
                scaling_factor = oe_cont.metadata.bit_volts[channel_index]
                decimated_samples /= scaling_factor
                decimated_samples = decimated_samples.astype(np.int16)

            # Write decimated continuous data using file writer abstraction
            file_writer.write_continuous_data(
                data=decimated_samples,
                channel_info=channel_info,
                sample_rate_hz=oe_metadata.sample_rate / config.downsampling_factor,
                start_time_ns=np.int64(oe_cont.timestamps[0] * 1e9),
                channel_name=f"{oe_metadata.stream_name}/{channel_name}/LFP",
                group_id=dh5_cont_id,
                calibration=np.array(np.float64(scaling_factor)),
            )

            dh5_cont_id += 1
            global_channel_index += 1
            ch_done += 1
            if on_channel:
                on_channel(ch_done, total_channels)

    # Add operation using file writer abstraction
    file_writer.add_operation(
        operation_name="Decimate raw data",
        tool_version=f"ephyscon.decimation (v{ephyscon.version.get_version_from_pyproject()})",
    )

    config.included_channel_names = included_channel_names
    return config
