"""Reference of the config files, generated from the code: every key of ``session.toml`` with its type, default
and description (from :class:`~ephysalis.recording_session.config.RecordingSessionConfig`), the layout of
``conversion_batch.toml`` (from :mod:`ephysalis.conversion_batch.config`), and a JSON Schema of ``session.toml``
for editors (e.g. VS Code with Even Better TOML / Taplo: ``#:schema ./session.schema.json`` as the first line).

    python -m ephysalis.config_reference docs     # docs/configuration.md, docs/schema/session.schema.json

``tests/test_config_reference.py`` checks that the files in ``docs/`` are current.
"""

from __future__ import annotations

import json
import re
import sys
import types
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ephysalis.recording_session.config import SCHEMA_VERSION, RecordingSessionConfig

MARKDOWN_FILE = "configuration.md"
JSON_SCHEMA_FILE = Path("schema") / "session.schema.json"


def _model_of(annotation: Any) -> type[BaseModel] | None:
    return annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None


def _strip_optional(annotation: Any) -> Any:
    if typing.get_origin(annotation) in (types.UnionType, typing.Union):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _type_name(annotation: Any) -> str:
    annotation = _strip_optional(annotation)
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return " / ".join(json.dumps(a) for a in typing.get_args(annotation))
    if origin in (types.UnionType, typing.Union):
        return " or ".join(_type_name(a) for a in typing.get_args(annotation))
    if origin is not None:
        args = ", ".join(_type_name(a) for a in typing.get_args(annotation))
        return f"{getattr(origin, '__name__', str(origin))}[{args}]"
    if annotation is typing.Any:
        return "any"
    return {str: "string", int: "integer", float: "float", bool: "boolean"}.get(annotation, getattr(annotation, "__name__", str(annotation)))


def _default(field: Any) -> str:
    if field.is_required():
        return "required"
    value = field.get_default(call_default_factory=True)
    if isinstance(value, BaseModel):
        return ""
    if isinstance(value, dict):
        value = {k: v.model_dump(mode="json") if isinstance(v, BaseModel) else v for k, v in value.items()}
    if isinstance(value, list):
        value = [v.model_dump(mode="json") if isinstance(v, BaseModel) else v for v in value]
    text = json.dumps(value, ensure_ascii=False)
    return f"`{text}`" if text not in ("null", '""', "[]", "{}") else ("none" if text == "null" else f"`{text}`")


def _markdown(text: str) -> str:
    """Docstring text as Markdown: one line, ``code`` -> `code`, :role:`x` -> `x`."""
    text = re.sub(r":[a-z]+:`~?([^`]+)`", r"`\1`", " ".join(text.split()))
    return text.replace("``", "`")


def _cell(text: str) -> str:
    return _markdown(text).replace("|", "\\|")


def _first_paragraph(model: type[BaseModel]) -> str:
    return _markdown((model.__doc__ or "").strip().split("\n\n")[0])


def _sections(model: type[BaseModel], header: str, path: str, note: str,
              out: list[tuple[str, str, list[tuple[str, str, str, str]]]]) -> None:
    """(header, note, rows) of ``model`` (a table at ``path``) and its sub-tables, depth first."""
    rows, below = [], []
    for name, field in model.model_fields.items():
        annotation = _strip_optional(field.annotation)
        origin, args = typing.get_origin(annotation), typing.get_args(annotation)
        key_path = f"{path}.{name}" if path else name
        description = field.description or ""
        if _model_of(annotation):
            below.append((annotation, f"[{key_path}]", key_path, description))
        elif origin is dict and len(args) == 2 and _model_of(args[1]):
            rows.append((name, f"table: name -> {args[1].__name__}", _default(field), description))
            below.append((args[1], f"[{key_path}.<name>]", f"{key_path}.<name>", description))
        elif origin is list and args and _model_of(args[0]):
            below.append((args[0], f"[[{key_path}]]", key_path, description))
        else:
            rows.append((name, _type_name(field.annotation), _default(field), description))
    out.append((header, note, rows))
    for sub_model, sub_header, sub_path, sub_note in below:
        first = _markdown(sub_note)
        first = first + "." if first and not first.endswith(".") else first
        note = " ".join(t for t in (first, _first_paragraph(sub_model)) if t)
        _sections(sub_model, sub_header, sub_path, note, out)


def session_toml_markdown() -> str:
    sections: list[tuple[str, str, list[tuple[str, str, str, str]]]] = []
    _sections(RecordingSessionConfig, "top level", "", "", sections)
    lines = [f"## session.toml (schema version {SCHEMA_VERSION})", "",
             "Per recording session; in a conversion batch the `[defaults]`, the session's file and its "
             "`session_config` are merged (tables key by key). A file without `schema_version` is taken as the "
             "current version; other versions are not read.", ""]
    for header, note, rows in sections:
        lines += [f"### `{header}`" if header != "top level" else "### top level", ""]
        if note:
            lines += [note, ""]
        if rows:
            lines += ["| key | type | default | description |", "|---|---|---|---|"]
            lines += [f"| `{k}` | {_cell(t)} | {_cell(d)} | {_cell(desc)} |" for k, t, d, desc in rows]
            lines.append("")
    return "\n".join(lines)


def conversion_batch_markdown() -> str:
    """The module docstring of :mod:`ephysalis.conversion_batch.config`: intro, the example as a TOML block, the
    rules."""
    from ephysalis.conversion_batch import config as batch

    doc = (batch.__doc__ or "").split("\n", 2)[2] if (batch.__doc__ or "").count("\n") >= 2 else ""
    intro, _, rest = doc.partition("::\n")
    lines, example = rest.split("\n"), []
    while lines and (lines[0].startswith("    ") or not lines[0].strip()):
        example.append(lines.pop(0)[4:])
    paragraphs = [_markdown(p) for p in "\n".join(lines).split("\n\n") if p.strip()]
    return "\n".join([f"## conversion_batch.toml (schema version {batch.CONVERSION_BATCH_SCHEMA_VERSION})", "",
                      _markdown(intro) + ":", "", "```toml", "\n".join(example).strip("\n"), "```", "",
                      "\n\n".join(paragraphs), ""])


def step_settings_markdown() -> str:
    """Every ``[steps.<step>]`` table of a conversion batch (:mod:`ephysalis.step_settings`)."""
    from ephysalis.step_settings import STEP_SETTINGS

    lines = ["## Step settings (`[steps.<step>]` of conversion_batch.toml)", "",
             "In a conversion batch every setting of a step that runs is required; on the command line they are the "
             "step's options (`ephysalis lfp --rate-hz 1000`, booleans as `--x` / `--no-x`), with these defaults.", ""]
    for step, model in STEP_SETTINGS.items():
        lines += [f"### `[steps.{step}]`", ""]
        if model.__dict__.get("__doc__"):  # the class's own docstring, not the base class's
            lines += [_first_paragraph(model), ""]
        if model.model_fields:
            lines += ["| key | type | default | description |", "|---|---|---|---|"]
            lines += [f"| `{name}` | {_cell(_type_name(field.annotation))} | {_cell(_default(field))} | "
                      f"{_cell(field.description or '')} |" for name, field in model.model_fields.items()]
        else:
            lines.append("No settings (an empty table runs the step).")
        lines.append("")
    return "\n".join(lines)


def reference_markdown() -> str:
    return "\n".join(["# Configuration reference", "",
                      "Generated by `python -m ephysalis.config_reference docs` from the code; do not edit. "
                      "JSON Schema of session.toml for editors: `schema/session.schema.json`.", "",
                      conversion_batch_markdown(), step_settings_markdown(), session_toml_markdown()])


def session_toml_json_schema() -> dict[str, Any]:
    schema = RecordingSessionConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = f"ephysalis session.toml (schema version {SCHEMA_VERSION})"
    # [aux_channels.digital]: an entry or the bit alone (the model's before-validator)
    digital = schema["$defs"]["AuxChannelMap"]["properties"]["digital"]
    digital["additionalProperties"] = {"anyOf": [digital["additionalProperties"],
                                                 {"type": "integer", "description": "the bit alone"}]}
    return schema


def write_config_reference(folder: Path) -> list[Path]:
    folder = Path(folder)
    markdown, schema = folder / MARKDOWN_FILE, folder / JSON_SCHEMA_FILE
    schema.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(reference_markdown(), encoding="utf-8", newline="\n")
    schema.write_text(json.dumps(session_toml_json_schema(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                      newline="\n")
    return [markdown, schema]


if __name__ == "__main__":
    for path in write_config_reference(Path(sys.argv[1] if len(sys.argv) > 1 else "docs")):
        print(path)

