"""Resumable Kilosort4: checkpoints after the long steps, and spike arrays allocated at full size.

A Kilosort4 run over a whole session takes hours and keeps no intermediate results. Inside
:func:`checkpoints`, the three long steps save their result to a folder and, when run again
with the same settings and data, load it instead of recomputing:

====================  =========================================  ===================
step                  Kilosort function                          share of runtime
====================  =========================================  ===================
``drift``             ``datashift.run``                          ~45 %
``spikedetect``       ``spikedetect.run`` (universal templates)  ~45 %
``extract``           ``template_matching.extract`` (learned)    ~10 %
====================  =========================================  ===================

Each checkpoint also holds the random number generator states, so a resumed run gives the
same result as an uninterrupted one.

Kilosort collects the spikes of these steps in arrays of 10**6 rows that it doubles when
full, which briefly holds the old and the doubled array. Inside :func:`checkpoints` the
arrays start at the spike count of the previous step plus a margin instead.

The steps are wrapped by replacing the module functions while the context is active, so
this works through ``spikeinterface.sorters.run_sorter``. Tested with Kilosort 4.1.7.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_KILOSORT_CAPACITY = 10**6  # initial number of rows of Kilosort's spike arrays
CAPACITY_MARGIN = 1.3  # preallocate this many times the spike count of the previous step


class CheckpointStore:
    """Checkpoint files of one Kilosort run; emptied when the settings or the data change."""

    def __init__(self, folder: Path, key: str):
        self.folder = folder
        self.key = key
        meta = folder / "checkpoints.json"
        if meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("key") != key:
            logger.info("kilosort checkpoints in %s are for other settings or data; discarding them", folder)
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({"key": key}), encoding="utf-8")

    def path(self, step: str) -> Path:
        return self.folder / f"{step}.pt"

    def has(self, step: str) -> bool:
        return self.path(step).exists()

    def save(self, step: str, ops: dict, result: Any) -> None:
        import torch

        tmp = self.path(step).with_suffix(".tmp")
        torch.save({"ops": ops, "result": result, "rng": _rng_state()}, tmp)
        tmp.replace(self.path(step))

    def load(self, step: str, ops: dict) -> Any:
        """Result of ``step``; updates ``ops`` in place (Kilosort's callers keep their own reference)."""
        import torch

        data = torch.load(self.path(step), weights_only=False)
        ops.update(data["ops"])
        _set_rng_state(data["rng"])
        return data["result"]


def run_key(ops: dict) -> str:
    """Hash of the Kilosort version, the settings and the data files (paths and sizes)."""
    import kilosort

    filenames = ops["filename"]  # a list of files in Kilosort >= 4.0.30
    files = [Path(f) for f in (filenames if isinstance(filenames, (list, tuple)) else [filenames])]
    content = json.dumps({
        "kilosort": kilosort.__version__,
        "settings": ops["settings"],
        "files": [(str(f), f.stat().st_size if f.exists() else None) for f in files],
    }, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


def _rng_state() -> dict:
    import torch

    return {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict) -> None:
    import torch

    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class _Preallocating:
    """Stand-in for the ``numpy``/``torch`` module of a Kilosort module whose ``zeros`` returns
    ``capacity`` rows instead of Kilosort's initial 10**6; everything else is passed through."""

    def __init__(self, module: Any, capacity: int):
        self._module = module
        self._capacity = capacity

    def zeros(self, shape: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(shape, tuple) and shape and shape[0] == _KILOSORT_CAPACITY:
            shape = (max(self._capacity, _KILOSORT_CAPACITY), *shape[1:])
        return self._module.zeros(shape, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


@contextlib.contextmanager
def _preallocate(module: Any, capacity: int | None) -> Iterator[None]:
    if capacity is None:
        yield
        return
    saved = module.np, module.torch
    module.np, module.torch = _Preallocating(saved[0], capacity), _Preallocating(saved[1], capacity)
    try:
        yield
    finally:
        module.np, module.torch = saved


@contextlib.contextmanager
def checkpoints(folder: Path) -> Iterator[None]:
    """Make Kilosort4 runs inside this context resumable from ``folder`` (see the module docs)."""
    from kilosort import datashift, spikedetect, template_matching

    originals = {
        (datashift, "run"): datashift.run,
        (spikedetect, "run"): spikedetect.run,
        (template_matching, "extract"): template_matching.extract,
    }
    state: dict[str, Any] = {"store": None, "in_drift": False, "n_spikes": None}

    def store(ops: dict) -> CheckpointStore:
        if state["store"] is None:
            state["store"] = CheckpointStore(folder, run_key(ops))
        return state["store"]

    def checkpointed(step: str, run: Callable[[], tuple], ops: dict, n_spikes: Callable[[tuple], int]) -> tuple:
        s = store(ops)
        if s.has(step):
            logger.info("kilosort: loading the %s step from %s", step, s.path(step))
            result = s.load(step, ops)
        else:
            result = run()
            s.save(step, ops, result)
        state["n_spikes"] = n_spikes(result)
        return result

    def capacity() -> int | None:
        return None if state["n_spikes"] is None else int(state["n_spikes"] * CAPACITY_MARGIN)

    def drift(ops: dict, bfile: Any, *args: Any, **kwargs: Any) -> tuple:
        def run() -> tuple:
            state["in_drift"] = True
            try:
                return originals[(datashift, "run")](ops, bfile, *args, **kwargs)
            finally:
                state["in_drift"] = False

        result = checkpointed("drift", run, ops, lambda r: 0 if r[1] is None else len(r[1]))
        return (ops, result[1])

    def detect(ops: dict, bfile: Any, *args: Any, **kwargs: Any) -> tuple:
        if state["in_drift"]:  # the drift step detects spikes too; it is checkpointed as a whole
            return originals[(spikedetect, "run")](ops, bfile, *args, **kwargs)

        def run() -> tuple:
            with _preallocate(spikedetect, capacity()):
                return originals[(spikedetect, "run")](ops, bfile, *args, **kwargs)

        st, tF, _ = checkpointed("spikedetect", run, ops, lambda r: len(r[0]))
        return st, tF, ops

    def extract(ops: dict, bfile: Any, *args: Any, **kwargs: Any) -> tuple:
        def run() -> tuple:
            with _preallocate(template_matching, capacity()):
                return originals[(template_matching, "extract")](ops, bfile, *args, **kwargs)

        st, tF, _ = checkpointed("extract", run, ops, lambda r: len(r[0]))
        return st, tF, ops

    datashift.run, spikedetect.run, template_matching.extract = drift, detect, extract
    try:
        yield
    finally:
        for (module, name), function in originals.items():
            setattr(module, name, function)
