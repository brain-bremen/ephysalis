"""docs/configuration.md and docs/schema/session.schema.json are generated from the code and must be current
(regenerate: python -m ephysalis.config_reference docs); the JSON Schema accepts a written session.toml."""

import json
import re
import tomllib
from pathlib import Path

from ephysalis.config_reference import JSON_SCHEMA_FILE, MARKDOWN_FILE, write_config_reference
from ephysalis.recording_session.config import (
    AuxChannelMap, RecordingSessionConfig, dumps_toml, save_recording_session_config,
)

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_reference_files_are_current(tmp_path):
    write_config_reference(tmp_path)
    for name in (MARKDOWN_FILE, JSON_SCHEMA_FILE):
        assert (DOCS / name).read_text(encoding="utf-8") == (tmp_path / name).read_text(encoding="utf-8"), \
            f"docs/{name} is out of date: python -m ephysalis.config_reference docs"
    text = (tmp_path / MARKDOWN_FILE).read_text(encoding="utf-8")
    assert "### `[aux_channels.digital.<name>]`" in text and "| `in_events` | boolean | `true` |" in text
    assert "[recording_sessions]" in text and not re.search(r"(?<!`)``(?!`)", text)  # no RST markup left


def test_json_schema_accepts_a_session_toml(tmp_path):
    jsonschema = __import__("pytest").importorskip("jsonschema")
    schema = json.loads((DOCS / JSON_SCHEMA_FILE).read_text(encoding="utf-8"))
    config = RecordingSessionConfig(session_id="s", subject_id="m",
                                    aux_channels=AuxChannelMap(analog={}, digital={"trial_start": 0}))
    save_recording_session_config(config, tmp_path / "session.toml")
    jsonschema.validate(tomllib.loads((tmp_path / "session.toml").read_text()), schema)
    jsonschema.validate({"session_id": "s", "subject_id": "m", "aux_channels": {"digital": {"frame": 13}}}, schema)
    assert dumps_toml({"a": 1}) == "a = 1\n"
