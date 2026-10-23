"""Shared fixtures for real-data integration tests."""
from pathlib import Path

import pytest
import open_ephys.analysis as oe
from open_ephys.analysis.formats.BinaryRecording import BinaryRecording
from dh5io import DH5File

from ephyscon.config import load_config_from_file
from ephyscon.convert_open_ephys_to_dh5 import convert_open_ephys_recording_to_dh5

_DATA_DIR = Path(__file__).parent / "data"
_SESSION_NAME = "Test_2026-03-12_16-32-08"

DATA_FOLDER = _DATA_DIR / _SESSION_NAME
GOLDEN_DH5_PATH = _DATA_DIR / f"{_SESSION_NAME}_exp1_rec1.dh5"
GOLDEN_CONFIG_PATH = _DATA_DIR / f"{_SESSION_NAME}_exp1_rec1.config.json"

skip_if_no_data = pytest.mark.skipif(
    not (DATA_FOLDER.exists() and GOLDEN_DH5_PATH.exists()),
    reason="Real data not available",
)


@pytest.fixture(scope="session")
def oe_recording() -> BinaryRecording:
    session = oe.Session(str(DATA_FOLDER))
    return session.recordnodes[0].recordings[0]


@pytest.fixture(scope="session")
def golden_dh5() -> DH5File:
    f = DH5File(str(GOLDEN_DH5_PATH))
    yield f
    f._file.close()


@pytest.fixture(scope="session")
def fresh_dh5_path(oe_recording, tmp_path_factory):
    """Re-run the conversion with the golden config and return path to the result."""
    config = load_config_from_file(GOLDEN_CONFIG_PATH)
    tmp_dir = tmp_path_factory.mktemp("fresh")
    session_name = str(tmp_dir / _SESSION_NAME)
    convert_open_ephys_recording_to_dh5(oe_recording, session_name, config)
    exp = oe_recording.experiment_index + 1
    rec = oe_recording.recording_index + 1
    return tmp_dir / f"{_SESSION_NAME}_exp{exp}_rec{rec}.dh5"
