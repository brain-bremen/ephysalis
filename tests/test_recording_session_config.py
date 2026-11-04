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


def test_schema_versions(tmp_path):
    """Only the current schema version is read; a file without one is taken as current. migrate() itself
    (for future versions): a chain of one migration per version step."""
    from ephysalis.recording_session.schema import migrate
    from ephysalis.recording_session.config import SCHEMA_VERSION

    path = tmp_path / "session.toml"
    path.write_text('session_id = "x"\nsubject_id = "y"\n[[runs]]\nname = "r"\nmetadata_plugin = "vstim"\n')
    config = load_recording_session_config(path)
    assert config.schema_version == SCHEMA_VERSION == 10 and config.runs[0].metadata_plugin == "vstim"
    assert {r for r, line in config.aux_channels.digital.items() if not line.in_events} == {"sync", "wheel"}
    save_recording_session_config(config, path)
    assert "schema_version = 10" in path.read_text()
    path.write_text('schema_version = 9\nsession_id = "x"\nsubject_id = "y"\n')
    with pytest.raises(ValueError, match="no migration from schema version 9"):
        load_recording_session_config(path)

    migrations = {1: lambda d: {k: v for k, v in d.items() if k != "a"} | {"b": d["a"]}, 2: lambda d: {**d, "b": d["b"] * 2}}
    assert migrate({"schema_version": 1, "a": 3}, 3, migrations) == {"schema_version": 3, "b": 6}
    assert migrate({"a": 3}, 2, migrations)["b"] == 3  # no version: oldest
    with pytest.raises(ValueError, match="no migration"):
        migrate({"schema_version": 1, "a": 1}, 4, migrations)
    with pytest.raises(ValueError, match="invalid"):
        migrate({"schema_version": "2"}, 2, migrations)


def test_digital_lines(tmp_path):
    """[aux_channels.digital]: one entry per line, or the bit alone; written with one aligned inline table per
    line."""
    import tomllib

    from ephysalis.recording_session.config import dumps_toml, recording_session_config_from_dict

    config = recording_session_config_from_dict({
        "session_id": "x", "subject_id": "y",
        "aux_channels": {"digital": {"trial_start": 0, "frame": {"bit": 13, "in_events": False},
                                     "lever_pressed": {"bit": 14, "active": "low", "description": "lever"}}}})
    digital = config.aux_channels.digital
    assert (digital["lever_pressed"].bit, digital["lever_pressed"].active, digital["lever_pressed"].description) == \
        (14, "low", "lever")
    assert not digital["frame"].in_events and digital["trial_start"].in_events and digital["trial_start"].bit == 0
    assert config.aux_channels.bits == {"trial_start": 0, "frame": 13, "lever_pressed": 14}

    path = tmp_path / "session.toml"
    save_recording_session_config(config, path)
    text = path.read_text()
    assert 'lever_pressed = { bit = 14, active = "low",  description = "lever", in_events = true }' in text
    assert 'trial_start   = { bit = 0,  active = "high", description = "",      in_events = true }' in text
    assert load_recording_session_config(path) == config
    data = {"defaults": {"aux_channels": {"digital": {"a": {"bit": 1}, "quoted key": {"bit": 2, "description": 'x "y"'}}}}}
    assert tomllib.loads(dumps_toml(data, prefix=("defaults",))) == data
    assert "[defaults.aux_channels.digital]" in dumps_toml(data, prefix=("defaults",))
