"""Schema versions of config files.

Every config file format carries its schema version (``schema_version`` in ``session.toml``).
Raise the version whenever the format
changes: fields added, removed or renamed, or a value's meaning changed. Files of a newer
version are rejected, so an older ephysalis cannot silently drop fields it does not know.
Older files are upgraded on loading by a chain of migrations, one per version step, that
turn the raw data of version ``n`` into version ``n + 1``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def migrate(
    data: dict[str, Any],
    current: int,
    migrations: Mapping[int, Migration],
    key: str = "schema_version",
    missing: int = 1,
    source: object = "config",
) -> dict[str, Any]:
    """Upgrade raw config ``data`` to schema version ``current``.

    ``migrations[n]`` turns version ``n`` into ``n + 1``. Data without ``key`` is taken as
    version ``missing``. Raises :class:`ValueError` for data newer than ``current``.
    """
    version = data.get(key, missing)
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"{source}: invalid {key} {version!r}")
    if version > current:
        raise ValueError(
            f"{source} has schema version {version}, newer than supported ({current}); update ephysalis"
        )
    data = dict(data)
    while version < current:
        if version not in migrations:
            raise ValueError(f"{source}: no migration from schema version {version}")
        data = migrations[version](data)
        version += 1
    data[key] = current
    return data
