"""The plugin interface: load-time checks of plugins against the Protocols (ephysalis.plugins)."""

import types

import pytest

from ephysalis import plugins
from ephysalis.plugins import (
    INPUT_FORMAT_CAPABILITIES,
    METADATA_PLUGIN_CAPABILITIES,
    PLUGIN_API_VERSION,
    ExtractsTrialTables,
    InputFormat,
    MetadataPlugin,
    NamesEventWords,
    ReadsDigitalLineEdges,
    plugin_problems,
    require,
)


def _module(**members):
    return types.SimpleNamespace(**{"name": "test", "plugin_api_version": PLUGIN_API_VERSION, **members})


def _trials(run, clock_alignment, auxiliary_signals):
    return None


def test_builtin_plugins_pass():
    for group, core, capabilities in ((plugins.INPUT_FORMAT_GROUP, InputFormat, INPUT_FORMAT_CAPABILITIES),
                                      (plugins.METADATA_PLUGIN_GROUP, MetadataPlugin, METADATA_PLUGIN_CAPABILITIES)):
        names = plugins.available(group)
        assert names, group
        for name in names:
            assert plugin_problems(plugins.get_plugin(group, name), core, capabilities) == [], name


def test_metadata_plugin_problems():
    good = _module(extract_trial_tables=_trials)
    assert plugin_problems(good, MetadataPlugin, METADATA_PLUGIN_CAPABILITIES) == []
    assert isinstance(good, ExtractsTrialTables) and not isinstance(good, NamesEventWords)

    nothing = _module()
    assert plugin_problems(nothing, MetadataPlugin, METADATA_PLUGIN_CAPABILITIES) == [
        "provides none of the metadata capabilities"]
    wrong = _module(extract_trial_tables=lambda run: None, plugin_api_version=0)
    problems = plugin_problems(wrong, MetadataPlugin, METADATA_PLUGIN_CAPABILITIES)
    assert any("plugin API version 0" in p for p in problems)
    assert "extract_trial_tables(run) instead of extract_trial_tables(run, clock_alignment, auxiliary_signals)" \
        in problems


def test_partly_provided_capability():
    fmt = _module(can_read_session_folder=lambda session_path: False, open_session=lambda session_path: None,
                  get_stream_recording=lambda run, kind, probe_id: None,
                  default_auxiliary_channel_roles=lambda run: None,
                  read_digital_line_edges=lambda run, lines: None)  # read_auxiliary_digital_line missing
    assert plugin_problems(fmt, InputFormat, INPUT_FORMAT_CAPABILITIES) == [
        "ReadsDigitalLineEdges only partly provided: missing read_auxiliary_digital_line"]
    with pytest.raises(TypeError, match="has no ReadsDigitalLineEdges"):
        require(fmt, ReadsDigitalLineEdges, "the test")
