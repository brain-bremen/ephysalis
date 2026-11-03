"""The stimulus files of the lab's MATLAB/Psychtoolbox stimulus code (``X:\\Codes\\ExperimentCode``).

Each run folder contains one ``<k>_stimulusStage<Name>`` folder per paradigm with

- ``params.mat``: cell array of parameter structs; every vector-valued field is a factor and
  conditions enumerate the full factorial design of each struct (see :func:`get_stim_from_index`)
- ``stimulation.mat`` (v7.3): ``stimResults.stimulusOrder`` (condition index per trial),
  ``screenInfo``, ``sessioninfo``, ...
- ``trialWiseInformation/``: per-trial files (eye tracking, host timestamps)
- ``TrialTimeCorrectionInfo.mat`` (optional): photodiode/host-time trial matching made by
  ``correctTrialMismatch_PreprocessScript``
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ephysalis.formats.matfile import read_mat, read_mat_path

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_PARADIGM_DIR_RE = re.compile(r"^(?P<index>\d+)_stimulusStage(?P<name>.+)$")
_STIM_FUNC_FIELD = "stimFunc"


def _field_length(value: Any) -> int:
    """MATLAB ``length`` of a parameter field (number of levels of the factor).

    Deviation from MATLAB: strings and structs count as a single level (MATLAB would count
    characters of a char field, which no paradigm relies on).
    """
    if isinstance(value, (str, dict)):
        return 1
    if isinstance(value, list):
        return len(value)
    arr = np.asarray(value)
    return 0 if arr.size == 0 else (1 if arr.ndim == 0 else max(arr.shape))


def _field_level(value: Any, i: int) -> Any:
    if isinstance(value, (str, dict)):
        return value
    if isinstance(value, list):
        return value[i]
    arr = np.asarray(value)
    level = arr.item() if arr.ndim == 0 else arr.ravel(order="F")[i]
    return level.item() if isinstance(level, np.generic) else level


def get_stim_from_index(params: list[dict[str, Any]], index: int) -> tuple[int, dict[str, Any]]:
    """Port of ``getStimFromIndex.m``: map a 1-based condition index to its parameters.

    Conditions are numbered consecutively over the parameter structs; within a struct they
    enumerate all combinations of the field levels in MATLAB ``ind2sub`` (column-major)
    order, i.e. the first field varies fastest. Returns ``(param_set, stim)`` with a 0-based
    ``param_set``.
    """
    offset = 0
    for param_set, p in enumerate(params):
        dims = [_field_length(v) for v in p.values()]
        n = int(np.prod(dims)) if dims else 0
        if offset < index <= offset + n:
            subs = np.unravel_index(index - offset - 1, dims, order="F")
            stim = {
                name: (value if name == _STIM_FUNC_FIELD else _field_level(value, int(sub)))
                for (name, value), sub in zip(p.items(), subs)
            }
            return param_set, stim
        offset += n
    raise IndexError(f"Condition index {index} out of range (1..{offset})")


def _stim_function_name(stim_func: Any) -> str | None:
    if isinstance(stim_func, dict):
        handle = stim_func.get("function_handle", stim_func)
        if isinstance(handle, dict):
            return handle.get("function")
    return stim_func if isinstance(stim_func, str) else None


def load_params(paradigm_dir: Path) -> list[dict[str, Any]]:
    """Load the parameter structs of a paradigm, preserving field order."""
    params_file = paradigm_dir / "params.mat"
    source = params_file if params_file.exists() else paradigm_dir / "stimulation.mat"
    params = read_mat(source, variables=["params"])["params"]
    return [params] if isinstance(params, dict) else list(params)


def load_stimulus_order(paradigm_dir: Path) -> np.ndarray:
    """Condition index per presented trial; trailing NaNs (aborted paradigm) are removed."""
    order = np.atleast_1d(
        np.asarray(read_mat_path(paradigm_dir / "stimulation.mat", "stimResults/stimulusOrder"), float)
    )
    nan = np.flatnonzero(np.isnan(order))
    if nan.size:
        order = order[: nan[0]]
    return order.astype(int)
