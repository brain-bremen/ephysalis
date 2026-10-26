"""Entry-point based plugin registry for input formats and stimulus systems.

A plugin that fails to import (e.g. an entry point left over from a removed reader, or a
missing optional dependency) is skipped with a warning, so the other plugins keep working."""

from __future__ import annotations

import logging
from functools import cache
from importlib.metadata import entry_points
from typing import Any

INPUT_FORMAT_GROUP = "ephyscon.input_formats"
STIMULUS_SYSTEM_GROUP = "ephyscon.stimulus_systems"


logger = logging.getLogger(__name__)


@cache
def _load_group(group: str) -> dict[str, Any]:
    plugins = {}
    for ep in entry_points(group=group):
        try:
            plugins[ep.name] = ep.load()
        except Exception as error:  # noqa: BLE001 (one broken plugin must not disable the others)
            logger.warning("plugin %s (%s) not loaded: %s", ep.name, ep.value, error)
    return plugins


def available(group: str) -> list[str]:
    return sorted(_load_group(group))


def get_plugin(group: str, name: str) -> Any:
    plugins = _load_group(group)
    if name not in plugins:
        raise KeyError(f"Unknown plugin {name!r} in {group}. Available: {sorted(plugins)}")
    return plugins[name]()
