"""Trials of VStim 1.x recorded with the lab's DAQ program (``.daq`` files, 2000-2006): the
``TD01`` trial descriptors in the DAQ file (trial and stimulus number, written when VStim
started the trial) joined with the VStim behavioural record (``.tdr``) of the day. Port of
``trials_from_daq`` / ``tdr_match`` / ``trial_recovery`` of the Fritz/Mini BIDS conversions and
the trial map of their 2006 conversion (``convert_v3.py``).

Per ``TD01`` record:

* start: the last rising edge of the trial start line (role ``trial_start`` in
  ``[aux_channels.digital]``, default line 1 = VStim interval 0) at or before the record;
  records without one are dropped;
* trial number, stimulus number: from the record; outcome, reaction time, reward: from the
  ``.tdr`` trial with that number (a different stimulus number there is logged);
* end: the first rising edge of the role ``trial_end`` after the record if that role is set
  (2006 setup: line 12); else the ``.tdr`` trial end (positive transition of virtual trigger
  11, VStim <= 1.34) after the start; else (VStim 1.1 ``.tdr`` without trigger times, or no
  ``.tdr``) the first outcome pulse + 335 ms (the ``.tdr`` end lies 330-339 ms after it);
  else the start;
* without a ``.tdr`` trial, the outcome comes from the first outcome pulse after the start
  (roles ``reward``, ``eye_error``, ``early_error``, ``late_error``; Hit, EyeErr, Early, Late;
  agrees with the ``.tdr`` in 99.9 % of the 46 282 trials of 2004-2005), else Undetermined (-1).

The ``.tdr``: VStim wrote one per day, and a session folder can hold runs of other days. The
candidates are the ``.tdr`` files of the session folder and of its sibling folders whose date
(4th line of the file) is the run's date (DAQ header), else all of the session folder; the one
with most trial numbers of the run (with the same stimulus number) is used.

Not handled: runs whose ``TD01`` records all carry the same trial number (DAQ program did not
receive the numbers, Fritz 2005-04-28): logged, the trials then get their outcome from the
trigger lines and no ``.tdr`` parameters.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from ephyscon.stimulus.base import Paradigm, StimulusSystem
from ephyscon.sync import AUX

if TYPE_CHECKING:
    from ephyscon.auxsignals import AuxSignals
    from ephyscon.session import Run
    from ephyscon.sync import RunSync

logger = logging.getLogger(__name__)

TRIAL_END_AFTER_OUTCOME_PULSE_MS = 335.0
OUTCOME_ROLES = {"reward": 1, "eye_error": 7, "early_error": 5, "late_error": 6}  # VStim TrialOutcome codes
OUTCOME_UNDETERMINED = -1
TRIAL_END_TRIGGER = 11  # VStim <= 1.34: virtual trigger 11 rises at the trial end


@cache
def _read_tdr(path: str) -> Any:
    from vstim.tdr import read_tdr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_tdr(Path(path))


@cache
def _tdr_trials(path: str) -> dict[int, Any]:
    return {int(t.trialNumber): t for t in _read_tdr(path).get_trials()}


def tdr_date(path: Path) -> date | None:
    """Date of a ``.tdr`` from its file-start header (4th line, ``dd.mm.yyyy<TAB>hh:mm:ss``)."""
    try:
        with open(path, encoding="latin-1") as file:
            for i, line in enumerate(file):
                if i == 3:
                    day = line.split()[0]
                    return date(int(day[6:10]), int(day[3:5]), int(day[0:2]))
    except (OSError, ValueError, IndexError):
        return None
    return None


def _candidates(run: Run) -> list[Path]:
    own = sorted(p for p in run.path.glob("*") if p.suffix.lower() == ".tdr")
    if run.start_time is None:
        return own
    day = run.start_time.date()
    near = own + sorted(p for p in run.path.parent.glob("*/*") if p.suffix.lower() == ".tdr" and p.parent != run.path)
    same_day = [p for p in near if tdr_date(p) == day]
    return same_day or own


def find_tdr(run: Run, descriptors: pd.DataFrame) -> Path | None:
    """The ``.tdr`` with most of the run's trial numbers (and the same stimulus numbers)."""
    best, best_score = None, 0
    pairs = list(zip(descriptors.trial_number.tolist(), descriptors.stimulus_number.tolist()))
    for path in _candidates(run):
        try:
            trials = _tdr_trials(str(path))
        except Exception as error:  # noqa: BLE001 (a damaged .tdr must not stop the search)
            logger.warning("%s: not read (%s)", path, error)
            continue
        score = sum(1 for no, stim in pairs if no in trials and trials[no].stimulusNumber == stim)
        if score > best_score:
            best, best_score = path, score
    return best


def tdr_trial_end_ms(trial: Any) -> float | None:
    """End of a trial in ms after its start (VStim <= 1.34): positive transition of virtual
    trigger 11, else the latest transition of triggers 0-11; None without trigger times."""
    positive, negative = trial.tPositiveTriggerTransitionMS, trial.tNegativeTriggerTransitionMS
    if positive is None or negative is None:
        return None
    if len(positive) > TRIAL_END_TRIGGER and positive[TRIAL_END_TRIGGER] >= 0:
        return float(positive[TRIAL_END_TRIGGER])
    candidates = [x for x in list(positive[:12]) + list(negative[:12]) if x >= 0]
    return float(max(candidates)) if candidates else None


def _rising(ttl: pd.DataFrame, bit: int) -> np.ndarray:
    return np.sort(ttl.aux_sample[(ttl.line == bit + 1) & (ttl.state == 1)].to_numpy())


def _descriptors(run: Run) -> pd.DataFrame | None:
    read = getattr(run.format, "read_trial_descriptors", None)
    return read(run) if read is not None else None


class VStimTD01System(StimulusSystem):
    name: ClassVar[str] = "vstim_td01"

    def can_handle(self, run: Run) -> bool:
        descriptors = _descriptors(run)
        return descriptors is not None and len(descriptors) > 0

    def discover(self, run: Run) -> list[Paradigm]:
        descriptors = _descriptors(run)
        if descriptors is None or not len(descriptors):
            return []
        path = find_tdr(run, descriptors)
        name = "trials"
        metadata: dict[str, Any] = {"tdr": None}
        if path is not None:
            header = _read_tdr(str(path)).headers[0]
            ini = Path(str(getattr(header, "iniFile", "") or "").replace("\\", "/")).stem
            name = ini or path.stem
            metadata = {"tdr": path.name, "vstim_version": getattr(header, "vstimVersion", None)}
        else:
            logger.warning("Run %s: no .tdr with its trial numbers found; outcomes from the trigger lines", run.name)
        return [Paradigm(name=name, run_name=run.name, index_in_run=0, system=self.name, path=path, metadata=metadata)]

    def load_conditions(self, paradigm: Paradigm) -> pd.DataFrame:
        """The ``.tdr`` trials (all of the day; :meth:`load_trials` keeps those of the run)."""
        rows = []
        if paradigm.path is not None:
            for k, trial in enumerate(_read_tdr(str(paradigm.path)).get_trials()):
                rows.append({"trial": k, "condition": int(trial.stimulusNumber), "vstim_trial": int(trial.trialNumber),
                             "outcome": int(trial.outcome.value), "outcome_name": trial.outcome.name})
        return pd.DataFrame(rows)

    def load_trials(self, run: Run, paradigm: Paradigm, sync: RunSync, aux: AuxSignals | None) -> pd.DataFrame:
        descriptors = _descriptors(run)
        assert descriptors is not None
        roles = aux.channels.digital if aux is not None else {}
        ttl = run.format.read_ttl_events(run)
        fs = run.format.aux_sampling_frequency(run)
        starts = _rising(ttl, roles.get("trial_start", 0))
        ends = _rising(ttl, roles["trial_end"]) if "trial_end" in roles else None
        outcome_pulses = sorted((s, code) for role, code in OUTCOME_ROLES.items() if role in roles
                                for s in _rising(ttl, roles[role]))
        pulse_samples = np.array([s for s, _ in outcome_pulses], dtype=np.int64)
        tdr = _tdr_trials(str(paradigm.path)) if paradigm.path is not None else {}
        numbers = descriptors.trial_number.to_numpy()
        if len(numbers) >= 3 and len(set(numbers.tolist())) == 1:
            logger.warning("Run %s: all TD01 records carry trial number %d (not received from VStim); trials "
                           "without .tdr parameters", run.name, int(numbers[0]))
            tdr = {}
        records = []
        for sample, number, stimulus in zip(descriptors.aux_sample, numbers, descriptors.stimulus_number):
            before = starts[starts <= sample]
            if before.size:
                records.append((int(before[-1]), int(sample), int(number), int(stimulus)))
        if len(records) < len(descriptors):
            logger.warning("Run %s: %d TD01 records without a trial start edge before them dropped",
                           run.name, len(descriptors) - len(records))
        rows = []
        for k, (start, sample, number, stimulus) in enumerate(records):
            following = records[k + 1][0] if k + 1 < len(records) else np.iinfo(np.int64).max
            pulse = int(np.searchsorted(pulse_samples, start))
            first_pulse = outcome_pulses[pulse] if pulse < len(outcome_pulses) and pulse_samples[pulse] < following else None
            trial = tdr.get(number)
            row: dict[str, Any] = {"trial": k, "condition": stimulus, "vstim_trial": number}
            if trial is not None:
                if int(trial.stimulusNumber) != stimulus:
                    logger.warning("Run %s: trial %d has stimulus %d in TD01 but %d in the .tdr", run.name, number,
                                   stimulus, int(trial.stimulusNumber))
                row.update(outcome=int(trial.outcome.value), outcome_name=trial.outcome.name, outcome_source="tdr",
                           time_sequence=int(trial.timeSequence), reaction_time_ms=float(trial.reactionTimeMS),
                           reward_ms=float(trial.rewardDurationMS) if trial.rewardDurationMS is not None else np.nan)
            elif first_pulse is not None:
                row.update(outcome=first_pulse[1], outcome_source="trigger lines")
            else:
                row.update(outcome=OUTCOME_UNDETERMINED, outcome_source="none")
            end, end_source = start, "start"
            if ends is not None:
                after = ends[ends > sample]
                if after.size and after[0] < following:
                    end, end_source = int(after[0]), "trial_end line"
            elif trial is not None and (end_ms := tdr_trial_end_ms(trial)) is not None:
                end, end_source = start + int(round(end_ms * fs / 1000)), "tdr"
            elif first_pulse is not None:
                end = int(first_pulse[0] + round(TRIAL_END_AFTER_OUTCOME_PULSE_MS * fs / 1000))
                end_source = "outcome pulse + 335 ms"
            row.update(start_sample=start, stop_sample=end, stop_source=end_source)
            rows.append(row)
        trials = pd.DataFrame(rows)
        if trials.empty:
            return trials
        if "outcome_name" not in trials:
            trials["outcome_name"] = ""
        from vstim.tdr import TrialOutcome

        names = {o.value: o.name for o in TrialOutcome}
        trials["outcome_name"] = [n if isinstance(n, str) and n else names.get(int(o), "Undetermined")
                                  for n, o in zip(trials.outcome_name, trials.outcome)]
        trials.insert(1, "start_time", sync.to_reference_time(AUX, trials.pop("start_sample").to_numpy(float)))
        trials.insert(2, "stop_time", sync.to_reference_time(AUX, trials.pop("stop_sample").to_numpy(float)))
        trials["tdr"] = paradigm.path.name if paradigm.path is not None else ""
        return trials
