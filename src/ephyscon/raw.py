import ephyscon.default_mappings as default
import ephyscon.version
from pydantic import BaseModel, Field
from open_ephys.analysis.recording import Continuous
from open_ephys.analysis.recording import Recording
from open_ephys.analysis.recording import ContinuousMetadata
from dhspec.cont import create_channel_info
import numpy as np
from ephyscon.file_writer import FileWriter


class RawConfig(BaseModel):
    model_config = {"extra": "ignore"}  # Ignore extra fields for backward compatibility

    split_channels_into_cont_blocks: bool = Field(
        default=True,
        title="Split channels into CONT blocks",
        description="Store each channel in its own CONT block. Required for most downstream processing",
    )
    cont_ranges: dict[default.ContGroups, tuple[int, int]] = Field(
        default_factory=lambda: default.DEFAULT_CONT_GROUP_RANGES.copy(),
        title="CONT block ranges",
        description="DH5 CONT block ID ranges per channel group",
    )
    included_channel_names: list[str] | None = Field(
        default=None,
        title="Included channels",
        description="Channel names to process. Leave empty to include all channels",
    )


def _create_cont_group_per_channel(
    oe_continuous: Continuous,
    file_writer: FileWriter,
    metadata: ContinuousMetadata,
    start_cont_id: int,
    first_global_channel_index: int,
    included_channel_names: list[str] | None = None,
):
    global_channel_index = first_global_channel_index

    assert metadata.channel_names is not None, "Channel names are not set in OE data."
    for channel_index, name in enumerate(metadata.channel_names):
        if included_channel_names is not None and name not in included_channel_names:
            continue

        dh5_cont_id = start_cont_id + channel_index

        channel_info = create_channel_info(
            GlobalChanNumber=global_channel_index,
            BoardChanNo=channel_index,
            ADCBitWidth=16,
            MaxVoltageRange=10.0,
            MinVoltageRange=10.0,
            AmplifChan0=0,
        )

        data = oe_continuous.samples[:, channel_index : channel_index + 1]

        # Write continuous data using file writer abstraction
        file_writer.write_continuous_data(
            data=data,
            channel_info=channel_info,
            sample_rate_hz=metadata.sample_rate,
            start_time_ns=np.int64(oe_continuous.timestamps[0] * 1e9),
            channel_name=name,
            group_id=dh5_cont_id,
            calibration=np.array(metadata.bit_volts[channel_index]),
        )

        global_channel_index += 1


def _create_cont_group_per_continuous_stream(
    oe_continuous: Continuous,
    file_writer: FileWriter,
    metadata: ContinuousMetadata,
    start_cont_id: int,
    last_global_channel_index: int = 0,
    included_channel_names: list[str] | None = None,
):
    raise NotImplementedError("Grouping channels into CONT blocks is not yet supported")

    # create a CONT group for the entire continuous stream
    dh5_cont_id = start_cont_id

    # TODO: This should be an array of channel info objects, one for each channel
    # but for now, we just create one channel info object for the entire stream
    channel_info = create_channel_info(
        GlobalChanNumber=last_global_channel_index,
        BoardChanNo=0,
        ADCBitWidth=16,
        MaxVoltageRange=10.0,
        MinVoltageRange=10.0,
        AmplifChan0=0,
    )

    # TODO: add streaming data if too large for memory
    data = oe_continuous.samples
    index = dh5io.cont.create_empty_index_array(1)

    dh5io.cont.create_cont_group_from_data_in_file(
        file=dh5file,
        cont_group_id=dh5_cont_id,
        data=data,
        index=index,
        sample_period_ns=np.int32(1.0 / metadata.sample_rate * 1e9),
        name=metadata.source_node_name,
        channels=channel_info,
        calibration=metadata.bit_volts,
    )


def process_oe_raw_data(
    config: RawConfig, recording: Recording, file_writer: FileWriter
) -> RawConfig:
    assert recording.continuous is not None, (
        "No continuous data found in the recording."
    )

    # continuous raw data
    global_channel_index = 0
    included_channel_names: list[str] = []
    for cont in recording.continuous:
        # cont: Continuous
        metadata: ContinuousMetadata = cont.metadata
        if config.included_channel_names is None and metadata.channel_names is not None:
            included_channel_names.extend(metadata.channel_names)

        # All continuous streams go to RAW
        cont_group = default.ContGroups.RAW
        group_range_start_index: int = config.cont_ranges[cont_group][0]
        start_cont_id = global_channel_index + group_range_start_index

        nSamples, nChannels = cont.samples.shape
        if config.split_channels_into_cont_blocks:
            _create_cont_group_per_channel(
                oe_continuous=cont,
                file_writer=file_writer,
                metadata=metadata,
                start_cont_id=start_cont_id,
                first_global_channel_index=global_channel_index,
                included_channel_names=config.included_channel_names,
            )
            global_channel_index += nChannels
        else:
            _create_cont_group_per_continuous_stream(
                oe_continuous=cont,
                file_writer=file_writer,
                metadata=metadata,
                start_cont_id=start_cont_id,
                included_channel_names=config.included_channel_names,
            )

    # update included channels in config
    config.included_channel_names = included_channel_names

    # Add operation using file writer abstraction
    file_writer.add_operation(
        operation_name="Write raw data",
        tool_version=f"ephyscon.raw (v{ephyscon.version.get_version_from_pyproject()})",
    )

    return config
