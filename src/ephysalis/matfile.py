"""Reading MATLAB ``.mat`` files (v5 and v7.3) into plain Python objects.

Unlike pymatreader/h5py, struct field order is preserved for v7.3 files (taken from the
``MATLAB_fields`` attribute instead of the alphabetical HDF5 key order). Several legacy
stimulus files rely on field order, e.g. ``getStimFromIndex`` enumerates conditions over
``struct2cell(params)``.

Conversion rules: structs -> dict, cell arrays -> list, numeric/logical arrays -> squeezed
numpy arrays (scalars -> Python scalars), char -> str.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def is_v73(path: str | Path) -> bool:
    with open(path, "rb") as f:
        f.seek(512)  # v7.3 files have a 512 byte MATLAB header before the HDF5 superblock
        return f.read(8) == _HDF5_SIGNATURE


def read_mat(
    path: str | Path, variables: Iterable[str] | None = None, missing_ok: bool = False
) -> dict[str, Any]:
    """Read (selected) variables of a MAT file.

    With ``missing_ok``, requested variables that are not in the file are left out of the
    result instead of raising a ``KeyError``.
    """
    path = Path(path)
    if is_v73(path):
        with h5py.File(path, "r") as f:
            names = list(variables) if variables is not None else [k for k in f if not k.startswith("#")]
            if missing_ok:
                names = [n for n in names if n in f]
            return {name: _convert(f[name], f) for name in names}

    from pymatreader import read_mat as _read_v5

    return _read_v5(str(path), variable_names=list(variables) if variables is not None else None)


def read_mat_path(path: str | Path, h5_path: str) -> Any:
    """Read one (possibly nested) item, e.g. ``"stimResults/stimulusOrder"``, from a v7.3 file."""
    with h5py.File(path, "r") as f:
        return _convert(f[h5_path], f)


def _matlab_class(obj: h5py.Dataset | h5py.Group) -> str | None:
    cls = obj.attrs.get("MATLAB_class")
    return cls.decode() if isinstance(cls, bytes) else cls


def _field_names(group: h5py.Group) -> list[str]:
    fields = group.attrs.get("MATLAB_fields")
    if fields is None:
        return [k for k in group.keys() if not k.startswith("#")]
    return [b"".join(np.asarray(f).tolist()).decode() for f in fields]


def _convert(obj: h5py.Dataset | h5py.Group, f: h5py.File) -> Any:
    cls = _matlab_class(obj)

    if isinstance(obj, h5py.Group):
        names = _field_names(obj)
        values = {name: obj[name] for name in names if name in obj}
        # struct arrays store each field as an array of references
        ref_fields = [v for v in values.values() if isinstance(v, h5py.Dataset) and v.dtype == h5py.ref_dtype]
        if cls == "struct" and ref_fields and len(ref_fields) == len(values) and ref_fields[0].size > 1:
            n = ref_fields[0].size
            return [
                {name: _convert(f[ds[()].ravel()[i]], f) for name, ds in values.items()}
                for i in range(n)
            ]
        return {name: _convert(value, f) for name, value in values.items()}

    if obj.attrs.get("MATLAB_empty", 0):
        return [] if cls == "cell" else ("" if cls == "char" else np.array([]))

    data = obj[()]
    if cls == "cell" or obj.dtype == h5py.ref_dtype:
        return [_convert(f[ref], f) for ref in np.asarray(data).ravel()]
    if cls == "char":
        arr = np.asarray(data)
        if arr.ndim == 2 and arr.shape[1] > 1 and arr.shape[0] > 1:
            # char matrix -> list of rows
            return ["".join(map(chr, row)) for row in arr.T]
        return "".join(map(chr, arr.ravel()))

    arr = np.asarray(data).T  # HDF5 stores MATLAB arrays with reversed dimensions
    if cls == "logical":
        arr = arr.astype(bool)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.item()
    return arr
