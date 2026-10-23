"""MATLAB/Psychtoolbox "stimulusStage" system (lab stimulus code in ``X:\\Codes\\ExperimentCode``).

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
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from ephyscon.matfile import read_mat, read_mat_path
from ephyscon.session import Run
from ephyscon.stimulus.base import Paradigm, StimulusSystem
from ephyscon.sync import AUX

if TYPE_CHECKING:
    from ephyscon.auxsignals import AuxSignals
    from ephyscon.sync import RunSync

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


class StimStageSystem(StimulusSystem):
    name: ClassVar[str] = "stimstage"

    @staticmethod
    def _paradigm_dirs(run: Run) -> list[tuple[int, str, Path]]:
        found = []
        for d in run.path.iterdir():
            match = _PARADIGM_DIR_RE.match(d.name)
            if d.is_dir() and match:
                found.append((int(match["index"]), match["name"], d))
        return sorted(found)

    def can_handle(self, run: Run) -> bool:
        return bool(self._paradigm_dirs(run))

    def discover(self, run: Run) -> list[Paradigm]:
        paradigms = []
        for i, (_, name, path) in enumerate(self._paradigm_dirs(run)):
            metadata: dict[str, Any] = {"folder": path.name}
            stimulation = path / "stimulation.mat"
            if stimulation.exists():
                metadata.update(
                    read_mat(stimulation, variables=["screenInfo", "sessioninfo"], missing_ok=True)
                )
                for key in ("stimulusCount", "repetition"):
                    metadata[key] = int(read_mat_path(stimulation, f"stimResults/{key}"))
            else:
                logger.warning("No stimulation.mat in %s", path)
            paradigms.append(
                Paradigm(
                    name=name,
                    run_name=run.name,
                    index_in_run=i,
                    system=self.name,
                    path=path,
                    metadata=metadata,
                )
            )
        return paradigms

    def load_conditions(self, paradigm: Paradigm) -> pd.DataFrame:
        assert paradigm.path is not None
        params = load_params(paradigm.path)
        order = load_stimulus_order(paradigm.path)

        rows = []
        for trial, condition in enumerate(order):
            param_set, stim = get_stim_from_index(params, int(condition))
            stim_func = stim.pop(_STIM_FUNC_FIELD, None)
            rows.append(
                {
                    "trial": trial,
                    "condition": int(condition),
                    "param_set": param_set,
                    "stimulus_function": _stim_function_name(stim_func),
                    **stim,
                }
            )
        # union of fields over parameter sets, missing fields become NaN (as in MATLAB)
        return pd.DataFrame(rows)

    def load_trials(
        self, run: Run, paradigm: Paradigm, sync: RunSync, aux: AuxSignals | None
    ) -> pd.DataFrame:
        """Trial times from the screen photodiode on the aux stream.

        Only runs with a single paradigm are supported. The MATLAB pipeline also split old
        multi-paradigm runs using the "paradigm" digital line; this is not ported yet.
        """
        if aux is None:
            raise ValueError(f"Run {run.name}: stimstage trial times need the aux (photodiode) stream")
        if len(self._paradigm_dirs(run)) > 1:
            raise NotImplementedError(
                f"Run {run.name} contains several paradigms; splitting by the paradigm line is not ported"
            )
        conditions = self.load_conditions(paradigm)
        pulses = aux.photodiode_pulses(n_trials=len(conditions))
        n = min(len(conditions), pulses.onsets.size)
        if pulses.onsets.size != len(conditions):
            logger.warning(
                "Run %s, paradigm %s: %d photodiode pulses but %d trials in stimulusOrder; "
                "keeping the first %d",
                run.name, paradigm.name, pulses.onsets.size, len(conditions), n,
            )
        trials = conditions.iloc[:n].copy()
        trials.insert(1, "start_time", sync.to_reference_time(AUX, pulses.onsets[:n]))
        trials.insert(2, "stop_time", sync.to_reference_time(AUX, pulses.offsets[:n]))
        trials.attrs["photodiode_mode"] = pulses.mode
        missing = trials["start_time"].isna()
        if missing.any():
            logger.warning("Run %s: %d trials without photodiode onset dropped", run.name, missing.sum())
            trials = trials[~missing]
        return trials.reset_index(drop=True)
