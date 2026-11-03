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


def _metadata_plugin_module(**members):
    return types.SimpleNamespace(**{"metadata_plugin_name": "test", "plugin_api_version": PLUGIN_API_VERSION,
                                    **members})


def _input_format_module(**members):
    return types.SimpleNamespace(**{"input_format_name": "test", "plugin_api_version": PLUGIN_API_VERSION,
                                    **members})


def _trials(run, clock_alignment, auxiliary_signals):
    return None


def test_builtin_plugins_pass():
    for group, core, capabilities in ((plugins.INPUT_FORMAT_GROUP, InputFormat, INPUT_FORMAT_CAPABILITIES),
                                      (plugins.METADATA_PLUGIN_GROUP, MetadataPlugin, METADATA_PLUGIN_CAPABILITIES)):
        plugin_names = plugins.available(group)
        assert plugin_names, group
        for plugin_name in plugin_names:
            assert plugin_problems(plugins.get_plugin(group, plugin_name), core, capabilities) == [], plugin_name


def test_metadata_plugin_problems():
    good = _metadata_plugin_module(extract_trial_tables=_trials)
    assert plugin_problems(good, MetadataPlugin, METADATA_PLUGIN_CAPABILITIES) == []
    assert isinstance(good, ExtractsTrialTables) and not isinstance(good, NamesEventWords)

    nothing = _metadata_plugin_module()
    assert plugin_problems(nothing, MetadataPlugin, METADATA_PLUGIN_CAPABILITIES) == [
        "provides none of the metadata capabilities"]
    wrong = _metadata_plugin_module(extract_trial_tables=lambda run: None, plugin_api_version=0)
    problems = plugin_problems(wrong, MetadataPlugin, METADATA_PLUGIN_CAPABILITIES)
    assert any("plugin API version 0" in p for p in problems)
    assert "extract_trial_tables(run) instead of extract_trial_tables(run, clock_alignment, auxiliary_signals)" \
        in problems


def test_partly_provided_capability():
    fmt = _input_format_module(can_read_session_folder=lambda session_path: False, open_session=lambda session_path: None,
                  get_stream_recording=lambda run, kind, probe_id: None,
                  default_auxiliary_channel_roles=lambda run: None,
                  read_digital_line_edges=lambda run, lines: None)  # read_auxiliary_digital_line missing
    assert plugin_problems(fmt, InputFormat, INPUT_FORMAT_CAPABILITIES) == [
        "ReadsDigitalLineEdges only partly provided: missing read_auxiliary_digital_line"]
    with pytest.raises(TypeError, match="has no ReadsDigitalLineEdges"):
        require(fmt, ReadsDigitalLineEdges, "the test")


def test_old_plugin_api_and_missing_name():
    """A version-1 plugin (``name`` instead of ``metadata_plugin_name``) is rejected with what is wrong."""
    old = types.SimpleNamespace(name="old", plugin_api_version=1, extract_trial_tables=_trials)
    problems = plugin_problems(old, MetadataPlugin, METADATA_PLUGIN_CAPABILITIES)
    assert "missing metadata_plugin_name" in problems
    assert any("plugin API version 1" in p for p in problems)
    assert plugins.plugin_name(_input_format_module()) == "test"
