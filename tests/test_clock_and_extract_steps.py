"""align-clocks and extract-trials / -events / -behavior: each writes only its own outputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import SPIKEGLX_CROP_RECORDING_SESSION
from ephysalis.cli.main import main
from ephysalis.timeline.events import COLUMNS, dh5_event_codes, load_run_events
from ephysalis.provenance import CLOCK_AND_EXTRACT_STEPS

RUN = "3_rfmappingFine_g0"


def test_steps_write_their_own_outputs(tmp_path):
    recording_session = str(SPIKEGLX_CROP_RECORDING_SESSION)
    common = ["--work-dir", str(tmp_path)]
    assert main(["align-clocks", recording_session, *common]) == 0
    clocks = json.loads((tmp_path / "clocks" / f"{RUN}.json").read_text())
    assert clocks["reference"] in clocks["streams"] and len(clocks["streams"]) >= 2
    assert all(s["sync_edges"] >= 2 and s["max_residual_ms"] < 1 for s in clocks["streams"].values())
    assert not (tmp_path / "trials").exists() and not (tmp_path / "events").exists()

    assert main(["extract-events", recording_session, *common]) == 0
    assert (tmp_path / "events").is_dir() and not (tmp_path / "trials").exists()
    assert list(pd.read_csv(tmp_path / "events" / f"{RUN}.csv").columns) == COLUMNS  # no DH5 codes

    assert main(["extract-trials", recording_session, *common]) == 0
    assert len(pd.read_csv(tmp_path / "trials" / f"{RUN}_RF.csv")) == 23
    assert main(["extract-behavior", recording_session, *common]) == 0
    assert {p.stem for p in (tmp_path / "provenance").glob("*.json")} == set(CLOCK_AND_EXTRACT_STEPS)


def test_dh5_event_codes(tmp_path):
    events = pd.DataFrame({"time": [0.1, 0.2, 0.3], "source": ["ttl", "ttl", "word"],
                           "line": [3, 3, None], "state": [1, 0, None], "word": [None, None, 7],
                           "name": ["a", "a", "x"]})
    events.to_csv(tmp_path / "events.csv", index=False)
    loaded = load_run_events(tmp_path / "events.csv")
    assert list(loaded.columns) == COLUMNS
    np.testing.assert_array_equal(dh5_event_codes(loaded, 1000), [3, -3, 1007])


def test_a_config_naming_runs_the_recording_does_not_have(tmp_path):
    """Their settings would not apply (e.g. include = false): the step stops and names the recording's runs."""
    import pytest

    from ephysalis.formats.input.base import open_recording_session
    from ephysalis.recording_session.config import (
        RunConfig,
        default_recording_session_config,
        save_recording_session_config,
    )

    config = default_recording_session_config(open_recording_session(SPIKEGLX_CROP_RECORDING_SESSION))
    config.runs.append(RunConfig(name="no_such_run", include=False))
    save_recording_session_config(config, tmp_path / "session.toml")
    with pytest.raises(SystemExit, match=f"runs not in the recording .*: no_such_run \\(its runs: .*{RUN}"):
        main(["align-clocks", str(SPIKEGLX_CROP_RECORDING_SESSION), "--config", str(tmp_path / "session.toml"),
              "--work-dir", str(tmp_path / "work")])
    assert not (tmp_path / "work" / "clocks").exists()


def test_an_entry_of_a_file_split_into_runs():
    """A DAQ file with pauses becomes runs <file>_part<k>: the entry of <file> applies to them (Fritz 2009-11-18:
    20091118_mapping_a, runs 20091118_mapping_a_part1 and _part2)."""
    from ephysalis.recording_session.config import RecordingSessionConfig, RunConfig, run_file_name

    config = RecordingSessionConfig(schema_version=10, session_id="s", subject_id="m", runs=[RunConfig(name="day_mapping_a", task="rfmapping"),
                                                              RunConfig(name="day_mapping_a_part2", include=False)])
    assert run_file_name("day_mapping_a_part1") == "day_mapping_a" and run_file_name("day_mapping_a") is None
    assert config.run_config("day_mapping_a_part1").task == "rfmapping"
    assert config.run_config("day_mapping_a_part2").include is False  # an entry of its own wins
    assert config.run_config("day_other_part1") is None
    assert config.task_of(["day_mapping_a_part1"]) == "rfmapping"

    import pytest

    from ephysalis.cli.recording_session_commands import check_config_runs
    from ephysalis.recording_session.model import RecordingSession, Run

    runs = [Run(name=n, path=Path("."), index=k, format=None) for k, n in enumerate(["day_mapping_a_part1", "day_mapping_a_part2"])]
    check_config_runs(RecordingSession(name="day", path=Path("."), format=None, runs=runs), config, None)  # no error
    config.runs.append(RunConfig(name="day_mapping_b"))
    with pytest.raises(SystemExit, match="runs not in the recording .*: day_mapping_b "):
        check_config_runs(RecordingSession(name="day", path=Path("."), format=None, runs=runs), config, None)


def test_pulse_lines(tmp_path):
    """A line with in_events = false (the sync line here; a 100 Hz frame line in the lab's setups) kept for its
    pulses: not in the events, but in the pulses of the exports."""
    from ephysalis.formats.input.base import open_recording_session
    from ephysalis.timeline.annotations import recording_session_pulses
    from ephysalis.recording_session.config import default_recording_session_config

    recording_session = str(SPIKEGLX_CROP_RECORDING_SESSION)
    assert main(["extract-events", recording_session, "--work-dir", str(tmp_path), "--pulse-lines", "sync"]) == 0
    events = pd.read_csv(tmp_path / "events" / f"{RUN}.csv")
    assert "sync" not in set(events["name"].dropna())
    session = open_recording_session(SPIKEGLX_CROP_RECORDING_SESSION)
    digital = default_recording_session_config(session).aux_channels.digital
    pulses, lines = recording_session_pulses(tmp_path, session.runs, {RUN: 0.0}, digital)
    sync = pulses[pulses.role == "sync"]
    assert len(sync) == 5 and (sync.stop_time[:4] > sync.start_time[:4]).all() and sync.stop_time.isna().iloc[4] and lines[digital["sync"].bit + 1] == "sync"
    assert main(["extract-events", recording_session, "--work-dir", str(tmp_path)]) == 0  # without: no file
    assert not (tmp_path / "events" / f"{RUN}.pulse_lines.npz").exists()
