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
with most trial numbers of the run minus stimulus mismatches is used, unless it has less than
half of the run's trials or more than 5 % mismatches (then the run has no ``.tdr``; as
``tdr_match`` of the BIDS conversions).

Frozen ``TD01`` (all records of a run carry the same trial number: the DAQ program did not
receive the numbers, Fritz 2005-04-28 runs a-d): the trials are matched to the day's ``.tdr``
by time (port of ``trial_recovery.match_by_time``). The trial starts are put on the PC clock
with the DAQ header time minus the day's DAQ-computer clock offset (median over the session's
other runs of that day with trial stamps, :mod:`ephysalis.legacy.pcclock`); the residual shift (within
30 s) is the one under which most starts fall into the 1 s window of a ``.tdr`` stamp
(``tAbsTrialStart``, NotStarted trials excluded), and each start takes, in order, the trial
whose window it is in. Accepted if one-to-one and increasing for >= 95 % of the trials, with the
trigger-line outcomes agreeing with the ``.tdr`` in >= 95 % of them; trial and stimulus numbers
then come from the ``.tdr`` (column ``trial_number_source``). Otherwise the trials keep the
frozen numbers and get their outcome from the trigger lines.

Run start (:meth:`VStimTD01System.run_start`): on the VStim PC's clock, fitted to the ``.tdr``
trial stamps (``pc_clock_time``; :mod:`ephysalis.legacy.pcclock`).
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

from ephysalis.runstart import RunStart
from ephysalis.stimulus.base import Paradigm, StimulusSystem
from ephysalis.sync import AUX

if TYPE_CHECKING:
    from ephysalis.auxsignals import AuxSignals
    from ephysalis.session import Run
    from ephysalis.sync import RunSync

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


def _readable_tdr(path: Path) -> dict[int, Any] | None:
    try:
        return _tdr_trials(str(path))
    except Exception as error:  # noqa: BLE001 (a damaged .tdr must not stop the search)
        logger.warning("%s: not read (%s)", path, error)
        return None


def find_tdr(run: Run, descriptors: pd.DataFrame) -> Path | None:
    """The ``.tdr`` of the run's trials (see the module docs), or None."""
    best, best_hits, best_mismatches = None, -1, 0
    pairs = list(zip(descriptors.trial_number.tolist(), descriptors.stimulus_number.tolist()))
    for path in _candidates(run):
        trials = _readable_tdr(path)
        if trials is None:
            continue
        hits = [(no, stim) for no, stim in pairs if no in trials]
        mismatches = sum(int(trials[no].stimulusNumber) != stim for no, stim in hits)
        if len(hits) - mismatches > best_hits - best_mismatches:
            best, best_hits, best_mismatches = path, len(hits), mismatches
    if best is not None and (best_hits < 0.5 * len(pairs) or best_mismatches > 0.05 * max(best_hits, 1)):
        logger.warning("Run %s: best .tdr %s rejected (%d of %d trials, %d stimulus mismatches)", run.name, best.name,
                       best_hits, len(pairs), best_mismatches)
        return None
    return best if best_hits > 0 else None


def frozen(descriptors: pd.DataFrame) -> bool:
    """All TD01 records (>= 3) carry the same trial number."""
    return len(descriptors) >= 3 and descriptors.trial_number.nunique() == 1


SHIFT_RANGE_S, SHIFT_STEP_S, STAMP_TOLERANCE_S = 30.0, 0.01, 0.1
MIN_MATCHED, MIN_OUTCOME_AGREEMENT = 0.95, 0.95


def stamp_seconds(trial: Any) -> int | None:
    """``tAbsTrialStart`` (hh:mm:ss) in s since midnight."""
    stamp = str(trial.tAbsTrialStart or "")
    if stamp.count(":") != 2:
        return None
    hh, mm, ss = map(int, stamp.split(":"))
    return hh * 3600 + mm * 60 + ss


def match_by_time(pc_s: np.ndarray, tdr: dict[int, Any], outcomes: list[int | None]) -> tuple[list[int | None], dict]:
    """``.tdr`` trial number (or None) per trial start (``pc_s``: approximate PC time of day in
    s); ``outcomes``: the trigger-line outcome per trial (for the check). Raises ValueError if
    the checks fail (see the module docs)."""
    from vstim.tdr import TrialOutcome

    candidates = sorted((n for n, t in tdr.items() if t.outcome != TrialOutcome.NotStarted and stamp_seconds(t) is not None),
                        key=lambda n: (stamp_seconds(tdr[n]), n))
    numbers = np.array(candidates)
    stamps = np.array([stamp_seconds(tdr[n]) for n in candidates], float)

    def assign(shift: float) -> tuple[np.ndarray, np.ndarray]:
        x = pc_s + shift
        j = np.searchsorted(stamps, x + STAMP_TOLERANCE_S, side="right") - 1
        ok = (j >= 0) & (x - stamps[np.clip(j, 0, None)] < 1 + STAMP_TOLERANCE_S)
        return j, ok

    if not len(stamps):
        raise ValueError("no started .tdr trial with a time stamp")
    shifts = np.arange(-SHIFT_RANGE_S, SHIFT_RANGE_S, SHIFT_STEP_S)
    score = np.array([assign(float(s))[1].sum() for s in shifts])
    shift = float(np.median(shifts[score == score.max()]))  # centre of the best plateau
    j, ok = assign(shift)
    used: set[int] = set()
    last = -1
    for k in range(len(j)):  # one-to-one and in order
        if ok[k] and (j[k] in used or j[k] <= last):
            ok[k] = False
        if ok[k]:
            used.add(int(j[k]))
            last = int(j[k])
    out = [int(numbers[j[k]]) if ok[k] else None for k in range(len(j))]
    agree = sum(o == int(tdr[n].outcome.value) for o, n in zip(outcomes, out) if n is not None)
    matched = int(ok.sum())
    report = {"n_trials": len(out), "n_matched": matched, "shift_s": round(shift, 2), "outcome_agreement": int(agree)}
    if matched < MIN_MATCHED * len(out) or agree < MIN_OUTCOME_AGREEMENT * max(matched, 1):
        raise ValueError(f"time matching with the .tdr failed: {report}")
    return out, report


def _header_start(run: Run) -> Any:
    """The DAQ header start (``start_time`` may already be the PC-clock start)."""
    return run.recorded_start_time or run.start_time


def _seconds_of_day(time: Any) -> float:
    return time.hour * 3600 + time.minute * 60 + time.second + time.microsecond / 1e6


def _records(run: Run, descriptors: pd.DataFrame, starts: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(trial start sample, TD01 sample, trial number, stimulus number) of the records with a start edge."""
    records = []
    for sample, number, stimulus in zip(descriptors.aux_sample, descriptors.trial_number, descriptors.stimulus_number):
        before = starts[starts <= sample]
        if before.size:
            records.append((int(before[-1]), int(sample), int(number), int(stimulus)))
    return records


def day_clock_offset(run: Run, start_bit: int) -> float | None:
    """Median DAQ header time minus PC time of the first sample over the other runs of the
    session recorded on the run's day that have trial stamps (s), or None."""
    from ephysalis.legacy.pcclock import pc_fit

    header = _header_start(run)
    if header is None:
        return None
    offsets = []
    for other in run.format.open_session(run.path).runs:
        if other.name == run.name or other.start_time is None or other.start_time.date() != header.date():
            continue
        descriptors = _descriptors(other)
        if descriptors is None or not len(descriptors) or frozen(descriptors):
            continue
        path = find_tdr(other, descriptors)
        if path is None:
            continue
        tdr = _tdr_trials(str(path))
        fs = other.format.aux_sampling_frequency(other)
        c, s = [], []
        for start, _, number, _ in _records(other, descriptors, _rising(other.format.read_ttl_events(other, {start_bit + 1}), start_bit)):
            stamp = stamp_seconds(tdr[number]) if number in tdr else None
            if stamp is not None:
                c.append(stamp)
                s.append(start / fs)
        fit = pc_fit(np.array(c, float), np.array(s, float))
        if fit is not None:
            offsets.append(_seconds_of_day(other.start_time) - fit["offset_s"])
    return float(np.median(offsets)) if offsets else None


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
        is_frozen = frozen(descriptors)
        if is_frozen:  # matched by time in load_trials: the day's .tdr
            day = run.start_time.date() if run.start_time is not None else None
            path = next((p for p in _candidates(run) if day is None or tdr_date(p) == day), None)
        else:
            path = find_tdr(run, descriptors)
        name = "trials"
        metadata: dict[str, Any] = {"tdr": None}
        if path is not None:
            header = _read_tdr(str(path)).headers[0]
            ini = Path(str(getattr(header, "iniFile", "") or "").replace("\\", "/")).stem
            name = ini or path.stem
            metadata = {"tdr": path.name, "vstim_version": getattr(header, "vstimVersion", None), "td01_frozen": is_frozen}
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

    @staticmethod
    def _match_frozen(run: Run, records: list, tdr: dict[int, Any], roles: dict[str, int], fs: float,
                      first_pulse: Any) -> tuple[str, dict[int, Any]]:
        """Trial numbers of a run with frozen TD01 records from the day's .tdr by time; returns
        (description, .tdr trials; empty if not matched) and replaces the numbers in ``records``."""
        frozen_number = records[0][2] if records else -1
        offset = day_clock_offset(run, roles.get("trial_start", 0))
        header = _header_start(run)
        if not tdr or offset is None or header is None:
            logger.warning("Run %s: all TD01 records carry trial number %d (not received from VStim), and %s; trials "
                           "without .tdr parameters", run.name, frozen_number,
                           "no .tdr of the day" if not tdr else "no other run of the day gives the PC clock")
            return f"TD01 frozen (all records trial {frozen_number}), not matched", {}
        pc_s = _seconds_of_day(header) - offset + np.array([r[0] for r in records], float) / fs
        outcomes = [p[1] if (p := first_pulse(k)) is not None else None for k in range(len(records))]
        try:
            numbers, report = match_by_time(pc_s, tdr, outcomes)
        except ValueError as error:
            logger.warning("Run %s: TD01 frozen; %s", run.name, error)
            return f"TD01 frozen (all records trial {frozen_number}), not matched: {error}", {}
        records[:] = [(start, sample, number if number is not None else old, stimulus)
                      for (start, sample, old, stimulus), number in zip(records, numbers)]
        logger.info("Run %s: TD01 frozen, trials matched to the .tdr by time: %s", run.name, report)
        return (f"TD01 frozen (all records trial {frozen_number}): matched to the .tdr by time ({report['n_matched']}/"
                f"{report['n_trials']} trials, shift {report['shift_s']:+.2f} s from the day's clock offset "
                f"{offset:.1f} s, trigger-line outcome agrees in {report['outcome_agreement']})"), tdr

    def run_start(self, run: Run, trials: dict[str, pd.DataFrame]) -> RunStart | None:
        """Start on the VStim PC's clock from the ``.tdr`` trial stamps (:mod:`ephysalis.legacy.pcclock`)."""
        from datetime import datetime, timedelta

        from ephysalis.legacy.pcclock import fit_method, pc_fit, trial_stamps

        header = _header_start(run)
        tables = [t for t in trials.values() if not t.empty]
        if header is None or not tables:
            return None
        fit = pc_fit(*trial_stamps(pd.concat(tables, ignore_index=True)))
        if fit is None:
            return None
        midnight = datetime(header.year, header.month, header.day)
        return RunStart(midnight + timedelta(seconds=fit["offset_s"]), fit_method(fit), fit)

    def load_trials(self, run: Run, paradigm: Paradigm, sync: RunSync, aux: AuxSignals | None) -> pd.DataFrame:
        descriptors = _descriptors(run)
        assert descriptors is not None
        roles = aux.channels.digital if aux is not None else {}
        needed = {roles.get("trial_start", 0) + 1} | {bit + 1 for role, bit in roles.items()
                                                     if role == "trial_end" or role in OUTCOME_ROLES}
        ttl = run.format.read_ttl_events(run, needed)
        fs = run.format.aux_sampling_frequency(run)
        starts = _rising(ttl, roles.get("trial_start", 0))
        ends = _rising(ttl, roles["trial_end"]) if "trial_end" in roles else None
        outcome_pulses = sorted((s, code) for role, code in OUTCOME_ROLES.items() if role in roles
                                for s in _rising(ttl, roles[role]))
        pulse_samples = np.array([s for s, _ in outcome_pulses], dtype=np.int64)
        tdr = _tdr_trials(str(paradigm.path)) if paradigm.path is not None else {}
        records = _records(run, descriptors, starts)
        if len(records) < len(descriptors):
            logger.warning("Run %s: %d TD01 records without a trial start edge before them dropped",
                           run.name, len(descriptors) - len(records))

        def first_pulse(k: int) -> tuple[int, int] | None:
            following = records[k + 1][0] if k + 1 < len(records) else np.iinfo(np.int64).max
            pulse = int(np.searchsorted(pulse_samples, records[k][0]))
            return outcome_pulses[pulse] if pulse < len(outcome_pulses) and pulse_samples[pulse] < following else None

        number_source = "TD01"
        if frozen(descriptors):
            number_source, tdr = self._match_frozen(run, records, tdr, roles, fs, first_pulse)
            if tdr:
                records = [(start, sample, number, int(tdr[number].stimulusNumber) if number in tdr else stimulus)
                           for start, sample, number, stimulus in records]
        rows = []
        for k, (start, sample, number, stimulus) in enumerate(records):
            pulse_k = first_pulse(k)
            following = records[k + 1][0] if k + 1 < len(records) else np.iinfo(np.int64).max
            trial = tdr.get(number)
            row: dict[str, Any] = {"trial": k, "condition": stimulus, "vstim_trial": number,
                                   "trial_number_source": number_source}
            if trial is not None:
                if int(trial.stimulusNumber) != stimulus:
                    logger.warning("Run %s: trial %d has stimulus %d in TD01 but %d in the .tdr", run.name, number,
                                   stimulus, int(trial.stimulusNumber))
                stamp = str(trial.tAbsTrialStart or "")  # PC time of day (ephysalis.legacy.pcclock)
                row.update(outcome=int(trial.outcome.value), outcome_name=trial.outcome.name, outcome_source="tdr",
                           pc_clock_time=stamp,
                           time_sequence=int(trial.timeSequence), reaction_time_ms=float(trial.reactionTimeMS),
                           reward_ms=float(trial.rewardDurationMS) if trial.rewardDurationMS is not None else np.nan)
            elif pulse_k is not None:
                row.update(outcome=pulse_k[1], outcome_source="trigger lines")
            else:
                row.update(outcome=OUTCOME_UNDETERMINED, outcome_source="none")
            end, end_source = start, "start"
            if ends is not None:
                after = ends[ends > sample]
                if after.size and after[0] < following:
                    end, end_source = int(after[0]), "trial_end line"
            elif trial is not None and (end_ms := tdr_trial_end_ms(trial)) is not None:
                end, end_source = start + int(round(end_ms * fs / 1000)), "tdr"
            elif pulse_k is not None:
                end = int(pulse_k[0] + round(TRIAL_END_AFTER_OUTCOME_PULSE_MS * fs / 1000))
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
