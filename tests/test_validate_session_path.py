import pytest
from pathlib import Path

from ephysalis.inspect import validate_session_path


def test_rejects_nonexistent_path(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        validate_session_path(tmp_path / "no_such_folder")


def test_rejects_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello")
    with pytest.raises(ValueError, match="Not a directory"):
        validate_session_path(f)


def test_rejects_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="does not appear to be an Open Ephys session"):
        validate_session_path(tmp_path)


def test_rejects_unrelated_folder(tmp_path):
    (tmp_path / "some_file.csv").write_text("data")
    (tmp_path / "subdir").mkdir()
    with pytest.raises(ValueError, match="does not appear to be an Open Ephys session"):
        validate_session_path(tmp_path)


def test_accepts_full_session(tmp_path):
    (tmp_path / "Record Node 101").mkdir()
    validate_session_path(tmp_path)  # should not raise


def test_accepts_experiment_folder_inside_record_node(tmp_path):
    exp = tmp_path / "Record Node 101" / "experiment1"
    exp.mkdir(parents=True)
    (exp / "recording1").mkdir()
    validate_session_path(exp)  # should not raise


def test_rejects_experiment_folder_without_record_node_ancestor(tmp_path):
    exp = tmp_path / "experiment1"
    exp.mkdir()
    (exp / "recording1").mkdir()
    with pytest.raises(ValueError, match="no Open Ephys session.*found in any parent"):
        validate_session_path(exp)
