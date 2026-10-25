"""VStim (the lab's C++ stimulus program, https://github.com/brain-bremen/VStim).

VStim writes one TDR file (trial data record) per recording and sends messages to Open Ephys
(``VSTIM: START RECORDING ... <path>.tdr``, ``TRIAL_START <n> TRIALTYPE ...``,
``TRIAL_END <n> ... OUTCOME ...``). It also drives trigger lines, which the aux stream records:
line 1 (``trial_start`` role) goes up at the start of every trial.

- Paradigm: one per run, named after the stimulus definition (``.ini``) in the TDR header.
- Conditions (:meth:`VStimSystem.load_conditions`): one row per TDR trial; ``condition`` is
  VStim's trial type (stimulus number), plus outcome, time sequence, reaction and reward time.
- Trial times (:meth:`VStimSystem.load_trials`): start = rising edge of the trial start line
  that precedes the trial's ``TRIAL_START`` message (the message comes over the network,
  ~0.1-0.5 s later); stop = start + the trial duration in the TDR. Without the line, the
  message times are used (network latency, warned).

The TDR file is searched under the name in the ``START RECORDING`` message, next to the
session and in its parent folder.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from ephyscon.session import Run
from ephyscon.stimulus.base import Paradigm, StimulusSystem
from ephyscon.sync import AUX

if TYPE_CHECKING:
    from ephyscon.auxsignals import AuxSignals
    from ephyscon.sync import RunSync

logger = logging.getLogger(__name__)

_START_RECORDING_RE = re.compile(r"START RECORDING .*?(?P<path>[^\\/]*\.tdr)\s*$", re.IGNORECASE)
_TRIAL_START_RE = re.compile(r"TRIAL_START\s+(?P<trial>\d+)")
MAX_MESSAGE_DELAY_S = 1.0  # TRIAL_START message after the trial start line


def _messages(run: Run) -> pd.DataFrame:
    try:
        messages = run.format.read_messages(run)
    except NotImplementedError:
        return pd.DataFrame({"aux_sample": [], "text": []})
    messages = messages[messages.text.str.contains("VSTIM:", regex=False)].copy()
    messages["text"] = messages.text.str.split("VSTIM:", n=1).str[1].str.strip()
    return messages.reset_index(drop=True)


def find_tdr(run: Run) -> Path | None:
    """The run's TDR file: named in the START RECORDING message, next to or above the session."""
    names = [m["path"] for text in _messages(run).text if (m := _START_RECORDING_RE.search(text))]
    folders = [run.path, run.path.parent, run.path.parent.parent]
    for name in names:
        for folder in folders:
            if (folder / name).exists():
                return folder / name
    if names:
        logger.warning("Run %s: TDR file %s not found in %s", run.name, names[0], [str(f) for f in folders])
    return None


def read_tdr(path: Path) -> Any:
    from vstim.tdr import read_tdr as _read

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unknown header", category=UserWarning)
        return _read(path)


class VStimSystem(StimulusSystem):
    name: ClassVar[str] = "vstim"

    def can_handle(self, run: Run) -> bool:
        return find_tdr(run) is not None

    def discover(self, run: Run) -> list[Paradigm]:
        path = find_tdr(run)
        if path is None:
            return []
        header = read_tdr(path).headers[0]
        ini = Path(str(getattr(header, "iniFile", "") or "").replace("\\", "/")).stem
        return [Paradigm(
            name=ini or path.stem, run_name=run.name, index_in_run=0, system=self.name, path=path,
            metadata={"tdr": path.name, "vstim_version": getattr(header, "vstimVersion", None),
                      "tdr_version": getattr(header, "tdrVersion", None), "ini": ini or None,
                      "refresh_rate": getattr(header, "refreshRate", None)},
        )]

    def load_conditions(self, paradigm: Paradigm) -> pd.DataFrame:
        assert paradigm.path is not None
        rows = []
        for k, trial in enumerate(read_tdr(paradigm.path).get_trials()):
            rows.append({
                "trial": k,
                "condition": int(trial.stimulusNumber),
                "vstim_trial": int(trial.trialNumber),
                "time_sequence": int(trial.timeSequence),
                "outcome": int(trial.outcome.value),
                "outcome_name": trial.outcome.name,
                "hit": bool(trial.wasHit),
                "reaction_time_ms": float(trial.reactionTimeMS),
                "reward_ms": float(trial.rewardDurationMS) if trial.rewardDurationMS is not None else np.nan,
                "duration_ms": float(trial.get_trial_duration()) if trial.get_trial_duration() is not None else np.nan,
            })
        return pd.DataFrame(rows)

    def load_trials(self, run: Run, paradigm: Paradigm, sync: RunSync, aux: AuxSignals | None) -> pd.DataFrame:
        trials = self.load_conditions(paradigm)
        messages = _messages(run)
        starts = messages[messages.text.str.match(_TRIAL_START_RE)].copy()
        starts["vstim_trial"] = starts.text.str.extract(_TRIAL_START_RE)["trial"].astype(int)
        message_sample = dict(zip(starts.vstim_trial, starts.aux_sample))

        edges = None
        if aux is not None and "trial_start" in aux.channels.digital:
            line = aux.digital("trial_start")
            edges = np.flatnonzero(~line[:-1] & line[1:]) + 1
            if edges.size == 0:
                logger.warning("Run %s: no edges on the trial start line; using the message times", run.name)
                edges = None
        fs = aux.sampling_frequency if aux is not None else run.format.aux_sampling_frequency(run)

        start_samples = np.full(len(trials), np.nan)
        source = np.full(len(trials), "", dtype=object)
        for k, vstim_trial in enumerate(trials.vstim_trial):
            sample = message_sample.get(vstim_trial)
            if sample is None:
                continue
            if edges is not None:
                before = edges[(edges <= sample) & (edges > sample - MAX_MESSAGE_DELAY_S * fs)]
                if before.size:
                    start_samples[k], source[k] = before[-1], "trigger"
                    continue
            start_samples[k], source[k] = sample, "message"
        if (source == "message").any():
            logger.warning("Run %s: %d trials timed by their (network) message instead of the trial start line",
                           run.name, int((source == "message").sum()))
        found = ~np.isnan(start_samples)
        if not found.all():
            logger.warning("Run %s: %d TDR trials without TRIAL_START message dropped", run.name, int((~found).sum()))
        trials = trials[found].copy()
        start = sync.to_reference_time(AUX, start_samples[found])
        trials.insert(1, "start_time", start)
        trials.insert(2, "stop_time", start + trials.duration_ms.fillna(0).to_numpy() / 1000)
        trials["start_source"] = source[found]
        return trials.reset_index(drop=True)
