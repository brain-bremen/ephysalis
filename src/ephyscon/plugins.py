"""Entry-point based plugin registry for input formats and stimulus systems."""

from __future__ import annotations

from functools import cache
from importlib.metadata import entry_points
from typing import Any

INPUT_FORMAT_GROUP = "ephyscon.input_formats"
STIMULUS_SYSTEM_GROUP = "ephyscon.stimulus_systems"


@cache
def _load_group(group: str) -> dict[str, Any]:
    return {ep.name: ep.load() for ep in entry_points(group=group)}


def available(group: str) -> list[str]:
    return sorted(_load_group(group))


def get_plugin(group: str, name: str) -> Any:
    plugins = _load_group(group)
    if name not in plugins:
        raise KeyError(f"Unknown plugin {name!r} in {group}. Available: {sorted(plugins)}")
    return plugins[name]()
