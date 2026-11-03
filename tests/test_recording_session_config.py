import pytest

from ephysalis.recording_session.config import (
    AuxChannelMap,
    default_recording_session_config,
    load_recording_session_config,
    save_recording_session_config,
)
from ephysalis.formats.input.base import open_recording_session

RUNS = [("3_rfmapping_g0", "2025-01-09T13:27:45"), ("5_DoseResponse_g0", "2025-01-09T13:58:53")]


def test_default_config_and_roundtrip(make_spikeglx_recording_session, tmp_path):
    recording_session = open_recording_session(make_spikeglx_recording_session(runs=RUNS))
    config = default_recording_session_config(recording_session)
    assert config.subject_id == "SOMxCD1_172"
    assert config.included_runs == ["3_rfmapping_g0", "5_DoseResponse_g0"]
    assert config.probes["imec0"].serial_number == recording_session.runs[0].probes["imec0"].serial_number
    assert config.aux_channels == AuxChannelMap()

    config.probes["imec0"].bad_channels = [3, 191]
    path = tmp_path / "session.toml"
    save_recording_session_config(config, path)
    assert load_recording_session_config(path) == config


def test_newer_schema_is_rejected(tmp_path):
    path = tmp_path / "session.toml"
    path.write_text('schema_version = 99\nsession_id = "x"\nsubject_id = "y"\n')
    with pytest.raises(ValueError, match="newer"):
        load_recording_session_config(path)


def test_legacy_aux_defaults():
    assert AuxChannelMap.legacy_default(8).analog["photodiode"] == 2
    assert AuxChannelMap.legacy_default(4).analog["photodiode"] == 1


def test_schema_migration(tmp_path):
    from ephysalis.recording_session.schema import migrate
    from ephysalis.recording_session.config import SCHEMA_VERSION

    path = tmp_path / "session.toml"
    path.write_text('schema_version = 1\nsession_id = "x"\nsubject_id = "y"\n[[runs]]\nname = "r"\n'
                    'stimulus_system = "vstim"\n')
    config = load_recording_session_config(path)
    assert config.schema_version == SCHEMA_VERSION == 8 and config.runs[0].name == "r"
    assert config.runs[0].metadata_plugin == "vstim"  # renamed in schema 8
    assert config.events.exclude_roles == ["sync", "wheel"]
    assert config.calibration.run is None and not config.line_noise.enabled
    save_recording_session_config(config, path)
    assert "schema_version = 8" in path.read_text()

    migrations = {1: lambda d: {k: v for k, v in d.items() if k != "a"} | {"b": d["a"]}, 2: lambda d: {**d, "b": d["b"] * 2}}
    assert migrate({"schema_version": 1, "a": 3}, 3, migrations) == {"schema_version": 3, "b": 6}
    assert migrate({"a": 3}, 2, migrations)["b"] == 3  # no version: oldest
    with pytest.raises(ValueError, match="no migration"):
        migrate({"schema_version": 1, "a": 1}, 4, migrations)
    with pytest.raises(ValueError, match="invalid"):
        migrate({"schema_version": "2"}, 2, migrations)
