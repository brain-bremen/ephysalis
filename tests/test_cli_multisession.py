"""Integration test for multi-session CLI conversion."""
import shutil
import sys
from unittest.mock import patch

import pytest

from conftest import DATA_FOLDER, GOLDEN_CONFIG_PATH, skip_if_no_data
from ephyscon.cli.main import main


@skip_if_no_data
def test_cli_converts_multiple_sessions(tmp_path):
    session1 = tmp_path / "SessionA"
    session2 = tmp_path / "SessionB"
    shutil.copytree(DATA_FOLDER, session1)
    shutil.copytree(DATA_FOLDER, session2)

    output = tmp_path / "output"

    with patch("sys.argv", [
        "ephyscon", "convert",
        str(session1), str(session2),
        "--output-folder", str(output),
        "--config", str(GOLDEN_CONFIG_PATH),
    ]):
        main()

    assert any(output.glob("SessionA*.dh5")), "No DH5 produced for SessionA"
    assert any(output.glob("SessionB*.dh5")), "No DH5 produced for SessionB"
