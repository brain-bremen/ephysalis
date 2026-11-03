"""Metadata plugin of the Freiburg Neuropixels setup: SpikeGLX recordings with the lab's
MATLAB/Psychtoolbox stimulus code. Trials from the stimulusStage files of each run
(:mod:`.stimulus`), config suggestions from the lab's MATLAB pipeline (:mod:`.pipeline`)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ephysalis.metadata.base import Paradigm
from ephysalis.metadata.psychtoolbox_neuropixels_freiburg.stimulus import (
    _PARADIGM_DIR_RE,
    _STIM_FUNC_FIELD,
    _stim_function_name,
    get_stim_from_index,
    load_params,
    load_stimulus_order,
    logger,
    read_mat,
    read_mat_path,
)
from ephysalis.session.model import Run
from ephysalis.timeline.clocks import AUX

if TYPE_CHECKING:
    from ephysalis.session.config import SessionConfig
    from ephysalis.session.model import Session
    from ephysalis.timeline.auxsignals import AuxSignals
    from ephysalis.timeline.clocks import RunSync


name = 'psychtoolbox_neuropixels_freiburg'
plugin_api_version = 1


def _paradigm_dirs(run: Run) -> list[tuple[int, str, Path]]:
    found = []
    for d in run.path.iterdir():
        match = _PARADIGM_DIR_RE.match(d.name)
        if d.is_dir() and match:
            found.append((int(match["index"]), match["name"], d))
    return sorted(found)


def _can_handle(run: Run) -> bool:
    return bool(_paradigm_dirs(run))


def add_session_config_from_metadata_files(session: Session, config: SessionConfig, metadata_file: Path | None
                                           ) -> list[str]:
    """Probes, bad channels and run selection of the MATLAB pipeline: the session's sheet of
    the lab workbook ``metadata_file`` (``CR_Experiments25.xlsx``; sheet = session folder name) and
    ``Depths.txt``, ``BrokenChannelsProbe<N>_0Index.txt``, ``folders2Combine.txt``."""
    from ephysalis.metadata.psychtoolbox_neuropixels_freiburg.pipeline import apply_pipeline_metadata

    return apply_pipeline_metadata(config, session, metadata_file)


def _discover(run: Run) -> list[Paradigm]:
    paradigms = []
    for i, (_, name, path) in enumerate(_paradigm_dirs(run)):
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
                system=name,
                path=path,
                metadata=metadata,
            )
        )
    return paradigms


def _load_conditions(paradigm: Paradigm) -> pd.DataFrame:
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


def _load_trials(run: Run, paradigm: Paradigm, sync: RunSync, aux: AuxSignals | None
) -> pd.DataFrame:
    """Trial times from the screen photodiode on the aux stream.

    Only runs with a single paradigm are supported. The MATLAB pipeline also split old
    multi-paradigm runs using the "paradigm" digital line; this is not ported yet.
    """
    if aux is None:
        raise ValueError(f"Run {run.name}: trial times need the aux (photodiode) stream")
    if len(_paradigm_dirs(run)) > 1:
        raise NotImplementedError(
            f"Run {run.name} contains several paradigms; splitting by the paradigm line is not ported"
        )
    conditions = _load_conditions(paradigm)
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


def extract_trial_tables(run: Run, clock_alignment: RunSync | None, auxiliary_signals: AuxSignals | None
                         ) -> dict[str, pd.DataFrame] | None:
    """One table per paradigm of the run (with times on the reference clock if ``clock_alignment``
    is given), the paradigm's details in the table's ``attrs``; None if the run is not from this setup."""
    if not _can_handle(run):
        return None
    tables = {}
    for paradigm in _discover(run):
        table = (_load_conditions(paradigm) if clock_alignment is None
                 else _load_trials(run, paradigm, clock_alignment, auxiliary_signals))
        table.attrs.update({k: v for k, v in paradigm.metadata.items() if k not in table.attrs})
        tables[paradigm.name] = table
    return tables
