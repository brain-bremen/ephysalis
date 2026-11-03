"""Deleting folders on network shares.

Some NFS servers return incomplete directory listings while entries of the directory are being
deleted (the listing's read position is invalidated), so ``shutil.rmtree`` and ``rm -rf`` miss
entries and fail with "Directory not empty". :func:`remove_folder` reads each directory's entries
completely before deleting any of them, and lists again until the directory is empty. ephysalis
deletes folders only with it, also before spikeinterface would overwrite one (``save(...,
overwrite=True)``, ``run_sorter(remove_existing_folder=True)``).
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path

REMOVE_ATTEMPTS = 20


def remove_folder(folder: str | Path) -> None:
    """Delete ``folder`` and everything in it (nothing if it does not exist; a file or link is unlinked)."""
    folder = Path(folder)
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        folder.unlink()
        return
    if not folder.exists():
        return
    for attempt in range(REMOVE_ATTEMPTS):
        entries = list(os.scandir(folder))  # the complete listing, before anything is deleted
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                remove_folder(entry.path)
            else:
                try:
                    os.unlink(entry.path)
                except FileNotFoundError:
                    pass
        try:
            os.rmdir(folder)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            if error.errno != errno.ENOTEMPTY or attempt == REMOVE_ATTEMPTS - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
