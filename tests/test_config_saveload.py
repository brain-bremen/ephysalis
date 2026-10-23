import os
from pathlib import Path
import tempfile
import pytest
import json


from ephyscon.config import (
    OpenEphysConversionConfig,
    RawConfig,
    DecimationConfig,
    EventPreprocessingConfig,
    TrialMapConfig,
    SpikeConfig,
    save_config_to_file,
    load_config_from_file,
    VERSION,
)
from ephyscon.mua import ContinuousMuaConfig


def make_sample_config():
    return OpenEphysConversionConfig(
        raw_config=RawConfig(),
        decimation_config=DecimationConfig(),
        event_config=EventPreprocessingConfig(),
        trialmap_config=TrialMapConfig(),
        spike_config=SpikeConfig(),
        continuous_mua_config=ContinuousMuaConfig()
    )


def test_save_and_load_config_roundtrip():
    config = make_sample_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(os.path.join(tmpdir, "test_config.json"))
        save_config_to_file(config_path, config)
        loaded_config = load_config_from_file(config_path)
        assert config == loaded_config
        assert loaded_config.raw_config == config.raw_config
        assert loaded_config.decimation_config == config.decimation_config
        assert loaded_config.event_config == config.event_config
        assert loaded_config.trialmap_config == config.trialmap_config


def test_save_config_creates_file():
    config = make_sample_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(os.path.join(tmpdir, "test_config.json"))
        save_config_to_file(config_path, config)
        assert os.path.exists(config_path)
        with open(config_path, "r") as f:
            data = f.read()
            assert '"raw_config"' in data
            assert '"decimation_config"' in data


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config_from_file(Path("nonexistent_config_file.json"))


def test_save_and_load_config_with_none_fields():
    config = OpenEphysConversionConfig(
        raw_config=None,
        decimation_config=None,
        event_config=None,
        trialmap_config=None,
        continuous_mua_config=None
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(os.path.join(tmpdir, "test_config_none.json"))
        save_config_to_file(config_path, config)
        loaded_config = load_config_from_file(config_path)
        assert loaded_config.raw_config is None
        assert loaded_config.decimation_config is None
        assert loaded_config.event_config is None
        assert loaded_config.trialmap_config is None
        assert loaded_config.spike_config is None


def test_load_config_with_newer_version(tmp_path):
    # Create a config dict with a newer version than supported
    config_data = {
        "raw_config": None,
        "decimation_config": None,
        "event_config": None,
        "trialmap_config": None,
        "continuous_mua_config": None,
        "config_version": VERSION + 1,
    }
    config_path = tmp_path / "newer_version_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    import ephyscon.config as config_mod

    with pytest.raises(ValueError) as excinfo:
        config_mod.load_config_from_file(config_path)
    assert "newer than supported version" in str(excinfo.value)


def test_old_config_with_stream_mapping_is_ignored(tmp_path):
    """Verify old configs with oe_processor_cont_group_map still load (field ignored)"""
    # Create a config dict with the old oe_processor_cont_group_map field
    config_data = {
        "raw_config": {
            "split_channels_into_cont_blocks": True,
            "cont_ranges": {
                "RAW": [1, 1600],
                "ANALOG": [1601, 2000],
                "LFP": [2001, 4000],
                "ESA": [4001, 6000],
                "AP": [6001, 8000],
            },
            "oe_processor_cont_group_map": {
                "PXIe-6341": "RAW",
                "USB-6343": "RAW",
            },
            "included_channel_names": None,
        },
        "decimation_config": None,
        "event_config": None,
        "trialmap_config": None,
        "continuous_mua_config": None,
        "config_version": VERSION,
    }
    config_path = tmp_path / "old_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    # Should load successfully (extra field ignored via model_config)
    loaded_config = load_config_from_file(config_path)
    assert loaded_config.raw_config is not None
    assert loaded_config.raw_config.split_channels_into_cont_blocks is True
    # The old field should not be present in the loaded config
    assert not hasattr(loaded_config.raw_config, "oe_processor_cont_group_map")
