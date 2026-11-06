"""Deleting and copying folders on network shares.

On some NFS shares (seen on europa, NFS 4.1) a folder's timestamp does not change when several
files are created in it within a short time, so the client keeps its cached listing of the folder:
listings miss files that exist, also in other processes, until the folder changes again. Then
``shutil.rmtree`` and ``rm -rf`` fail with "Directory not empty", and spikeinterface does not find
extensions it has just written. :func:`fresh_listing` sets the folder's timestamp before listing
it, which makes the client read the listing again; only for folders ephysalis writes itself (work
dirs, scratch dirs), never for source data. ephysalis deletes folders only with
:func:`remove_folder`, also before spikeinterface would overwrite one, and copies with
:func:`copy_folder`.
"""

from __future__ import annotations

import errno
import os
import shutil
import time
from pathlib import Path

REMOVE_ATTEMPTS = 20


def fresh_listing(folder: str | Path) -> list[os.DirEntry[str]]:
    """The entries of ``folder``, after setting its timestamp so that the NFS client lists it again (a folder
    ephysalis wrote; never a source folder)."""
    os.utime(folder)
    return list(os.scandir(folder))


def glob_in_work_folder(folder: str | Path, pattern: str) -> list[Path]:
    """``sorted(folder.glob(pattern))`` of a folder ephysalis writes (work dir, scratch dir), with a fresh
    listing (:func:`fresh_listing`); an empty list if the folder does not exist."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    os.utime(folder)
    return sorted(folder.glob(pattern))


def run_files(folder: str | Path, run_name: str, suffix: str, run_names: tuple[str, ...] | list[str] = ()) -> list[Path]:
    """The files ``<run_name>_<anything><suffix>`` of a run in ``folder`` (:func:`glob_in_work_folder`), without
    those of the other runs in ``run_names`` whose name starts with ``<run_name>_`` (runs ``task`` and
    ``task_aftercrash``: ``task_*.csv`` also matches ``task_aftercrash_protocol.csv``)."""
    longer = [n for n in run_names if n != run_name and n.startswith(run_name + "_")]
    return [p for p in glob_in_work_folder(folder, f"{run_name}_*{suffix}")
            if not any(p.name.startswith(n + "_") for n in longer)]


def copy_folder(source: str | Path, destination: str | Path) -> None:
    """Copy the folder ``source`` to ``destination`` (which must not exist), then refresh the listing of every
    copied folder so that readers see all files at once."""
    source, destination = Path(source), Path(destination)
    shutil.copytree(source, destination)
    for folder, _, _ in os.walk(source):  # the source's listing (local disk) is complete
        os.utime(destination / Path(folder).relative_to(source))


def remove_folder(folder: str | Path) -> None:
    """Delete ``folder`` and everything in it (nothing if it does not exist; a file or link is unlinked)."""
    folder = Path(folder)
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        folder.unlink()
        return
    if not folder.exists():
        return
    for attempt in range(REMOVE_ATTEMPTS):
        entries = fresh_listing(folder)  # the complete listing, before anything is deleted
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
