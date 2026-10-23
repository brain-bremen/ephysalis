import logging
from collections.abc import Callable
from pathlib import Path

from open_ephys.analysis.recording import Recording
from open_ephys.analysis.session import Session

from ephyscon.config import (
    DecimationConfig,
    EventPreprocessingConfig,
    OpenEphysConversionConfig,
    RawConfig,
    TrialMapConfig,
    ContinuousMuaConfig,
    save_config_to_file,
)
from ephyscon.decimation import decimate_raw_data
from ephyscon.events import process_oe_events
from ephyscon.raw import process_oe_raw_data
from ephyscon.trialmap import process_oe_trialmap
from ephyscon.mua import extract_continuous_mua
from ephyscon.version import get_version_from_pyproject
from ephyscon.file_writer import create_file_writer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def convert_open_ephys_recording_to_dh5(
    recording: Recording,
    session_name: str,
    config: OpenEphysConversionConfig | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
):
    assert recording.continuous is not None, (
        "No continuous data found in the recording."
    )

    if len(recording.continuous) == 0:
        raise ValueError(
            "No continuous data found in the recording. This is not supported."
        )

    board_names = [
        f"{cont.metadata.source_node_name}:{cont.metadata.source_node_id}"
        for cont in recording.continuous
    ]

    if config is None:
        config = OpenEphysConversionConfig(
            raw_config=None,  # RawConfig(split_channels_into_cont_blocks=True),
            decimation_config=DecimationConfig(),
            event_config=EventPreprocessingConfig(network_events_offset=1000),
            trialmap_config=TrialMapConfig(),
            continuous_mua_config=ContinuousMuaConfig(),
        )

    # Determine output filename extension based on format
    file_extension = ".dh5" if config.output_format == "dh5" else ".nwb"
    output_filename = f"{session_name}_exp{recording.experiment_index + 1}_rec{recording.recording_index + 1}{file_extension}"

    logger.info(
        f"Start converting OpenEphys recording from {recording.directory} to {output_filename} "
        f"(format: {config.output_format}) using ephyscon v{get_version_from_pyproject()}"
    )

    # Create format-specific file writer
    file_writer = create_file_writer(
        filename=output_filename,
        output_format=config.output_format,
        board_names=board_names,
        dh5_options=config.dh5_output_options,
        nwb_options=config.nwb_output_options,
    )

    # --- progress setup ---
    n_ch = sum(c.metadata.num_channels for c in (recording.continuous or []))
    _STEP_UNITS = [
        ("Raw",       1     if config.raw_config            is not None else 0),
        ("Events",    1     if config.event_config          is not None else 0),
        ("Trial map", 1     if config.trialmap_config       is not None else 0),
        ("LFP",       n_ch  if config.decimation_config     is not None else 0),
        ("MUA",       n_ch  if config.continuous_mua_config is not None else 0),
        ("Spikes",    1     if config.spike_config          is not None else 0),
    ]
    total_units = sum(u for _, u in _STEP_UNITS)
    cum_done = 0

    def _report(step_name: str, done: int, total: int) -> None:
        nonlocal cum_done
        cum_done += 1
        if on_progress:
            on_progress(step_name, min(cum_done, total_units), total_units)

    def _step_done(step_name: str) -> None:
        _report(step_name, 1, 1)

    # --- steps ---
    if config.raw_config is not None:
        config.raw_config = process_oe_raw_data(config.raw_config, recording, file_writer)
        _step_done("Raw")

    if config.event_config is not None:
        config.event_config = process_oe_events(
            config.event_config, recording=recording, file_writer=file_writer
        )
        _step_done("Events")

    if config.trialmap_config is not None:
        config.trialmap_config = process_oe_trialmap(
            config.trialmap_config, recording=recording, file_writer=file_writer
        )
        _step_done("Trial map")

    if config.decimation_config is not None:
        config.decimation_config = decimate_raw_data(
            config.decimation_config, recording=recording, file_writer=file_writer,
            on_channel=lambda done, total: _report("LFP", done, total),
        )

    if config.continuous_mua_config is not None:
        decimation_config = config.decimation_config
        if decimation_config is None:
            decimation_config = DecimationConfig()

        config.continuous_mua_config = extract_continuous_mua(
            config=config.continuous_mua_config,
            decimation_config=decimation_config,
            recording=recording,
            file_writer=file_writer,
            on_channel=lambda done, total: _report("MUA", done, total),
        )

    # Close the file writer
    file_writer.close()

    config_filename = Path(
        f"{session_name}_exp{recording.experiment_index + 1}_rec{recording.recording_index + 1}.config.json"
    )
    save_config_to_file(config_filename, config)

    logger.info(
        f"Finished converting OpenEphys recording from {recording.directory} to {output_filename}"
    )
    # report resulting file size in a human readable format


def get_output_paths(
    session_path: Path,
    output_folder: Path | None,
    output_format: str,
) -> list[Path]:
    """Return the output file paths that would be created for a session, without converting."""
    folder = output_folder or session_path.parent
    session = Session(str(session_path))
    session_name = str(folder / session_path.name)
    ext = f".{output_format}"
    paths = []
    for node in session.recordnodes:
        for recording in node.recordings:
            fname = (
                f"{session_name}"
                f"_exp{recording.experiment_index + 1}"
                f"_rec{recording.recording_index + 1}"
                f"{ext}"
            )
            paths.append(Path(fname))
    return paths


def convert_open_ephys_session(
    session_path: Path,
    output_folder: Path | None = None,
    config: OpenEphysConversionConfig | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """Convert all recordings in a single Open Ephys session to DH5."""
    folder = output_folder or session_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    session = Session(str(session_path))
    session_name = str(folder / session_path.name)
    for node in session.recordnodes:
        for recording in node.recordings:
            convert_open_ephys_recording_to_dh5(
                recording=recording,
                session_name=session_name,
                config=config,
                on_progress=on_progress,
            )


def convert_open_ephys_sessions(
    session_paths: list[Path],
    output_folder: Path | None = None,
    config: OpenEphysConversionConfig | None = None,
) -> None:
    """Convert all recordings across multiple Open Ephys sessions to DH5."""
    for session_path in session_paths:
        convert_open_ephys_session(session_path, output_folder=output_folder, config=config)
