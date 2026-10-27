# VStim 1.x trials (`TD01` and `.tdr`)

With the DAQ program (2000–2006) and MC_Rack (2010–2011), the stimulus PC running VStim 1.x sent
the trial and stimulus number of each trial to the acquisition system:

- **DAQ files:** as `TD01` records (see {doc}`daq`);
- **MC_Rack files:** as trigger events with trial synchronisation (see {doc}`mcd`).

VStim also wrote its own record of the day, the `.tdr` file. This is a text file read by the
`vstim` package (`vstim.tdr`). It holds the trial number, stimulus number, outcome, reaction time,
reward, the trigger transition times and the time of day of each trial start (`tAbsTrialStart`,
whole seconds).

The `vstim_td01` stimulus plugin joins the two. It ports `trials_from_daq`, `tdr_match` and
`trial_recovery` of the Fritz/Mini BIDS conversions, and their trial map of 2006
(`convert_v3.py`). On Fritz 2005-03-30 and 2005-04-28, trials and run starts equal the BIDS
dataset's trial maps and `scans.tsv`.

## Trigger lines

The roles of the trigger lines differ between setups and are set in `session.toml` (bit = line
− 1):

```toml
[aux_channels.digital]       # 2004-2005
trial_start = 0              # line 1: VStim interval 0
eye_error = 10               # lines 11-14: outcome pulses
early_error = 11
late_error = 12
reward = 13
lever_pressed = 14
# 2006 setup: trial_start = 0, trial_end = 11 (line 12), reward = 15
```

## Trials

For each trial descriptor:

- **Start:** the last rising edge of the `trial_start` line at or before the descriptor.
  Descriptors without such an edge are dropped.
- **Trial and stimulus number:** from the descriptor. Outcome, reaction time and reward come from
  the `.tdr` trial with that number, and a different stimulus number there is logged.
- **End:** the first of these that exists:
  1. the first rising edge of `trial_end` after the descriptor, if that role is set (2006);
  2. the `.tdr` trial end: the rising transition of virtual trigger 11 (VStim ≤ 1.34);
  3. the first outcome pulse + 335 ms (the `.tdr` end lies 330–339 ms after it); this covers
     VStim 1.1 `.tdr` files without trigger times;
  4. the start.
- **Outcome without a `.tdr` trial:** the first outcome pulse after the start (`reward` → Hit,
  `eye_error` → EyeErr, `early_error` → Early, `late_error` → Late), else Undetermined (−1). This
  agrees with the `.tdr` in 99.9 % of the 46 282 trials of 2004–2005.

## Finding the `.tdr`

VStim wrote one `.tdr` per day, and a session folder can hold runs of other days.
- **Candidates:** the `.tdr` files of the session folder and of its sibling folders whose date
  (4th line of the file) is the run's date. If there are none, all `.tdr` files of the session
  folder are used.
- **Choice:** the candidate with the most of the run's trial numbers, minus stimulus mismatches.
- **Rejection:** the run has no `.tdr` if the best candidate has less than half of the run's
  trials or more than 5 % mismatches.

## Frozen `TD01`

In some runs all `TD01` records carry the same trial number, because the DAQ program did not
receive the numbers (Fritz 2005-04-28, runs a–d). These trials are matched to the day's `.tdr`
by time:

1. **Clock offset of the day:** the DAQ computer's clock ran a constant few minutes ahead of the
   VStim PC on a given day. The offset is the median over the session's other runs of that day
   that have trial stamps (see below).
2. **Trial starts on the PC clock:** the DAQ header time minus that offset, plus each start's
   time in the run.
3. **Residual shift:** within ±30 s, the shift under which most starts fall into the 1 s window of
   a `.tdr` stamp. NotStarted trials are excluded.
4. **Assignment:** each start takes, in order, the trial whose window it falls in.

The match is accepted if two conditions hold:
- it is one-to-one and increasing for at least 95 % of the trials;
- the trigger-line outcomes agree with the `.tdr` in at least 95 % of them.

Trial and stimulus numbers then come from the `.tdr`, which the column `trial_number_source`
records. Otherwise the trials keep the frozen numbers and get their outcome from the trigger
lines.

## Run start on the PC clock

The DAQ header gives a file's start to 1 s, on the DAQ computer's clock. The offsets to the VStim
PC were 335 s on 2004-07-22, 398 s on 2005-01-06 and 885 s on 2005-08-16.

With the `.tdr` stamps, `pcclock.pc_fit` fits *PC = offset + (1 + ppm · 10⁻⁶) · t*, where *t* is
the trial start on the run's clock:
1. **Windows:** for every candidate rate (±300 ppm in 0.5 ppm steps), the trials' 1 s stamp
   windows are intersected.
2. **Rate:** 35 ppm (the median of the well-constrained runs) if the data allow it, else the rate
   with the widest common window.
3. **Offset:** the centre of that window. The window's width, 1–60 ms, is the precision.
4. **Jitter:** if no rate gives a common window (the stamps jitter by about 0.2 s in some 2004
   runs), the windows are widened by up to 0.5 s.

`ephysalis timing` stores this start in `<work>/run_start/<run>.json`. It places the runs of a
session file and becomes the BIDS `acq_time`, and the method text goes into `scans.tsv`.

## API

```{eval-rst}
.. autoclass:: ephysalis.legacy.vstim_td01.VStimTD01System
   :members: load_trials, run_start

.. autofunction:: ephysalis.legacy.vstim_td01.find_tdr

.. autofunction:: ephysalis.legacy.vstim_td01.match_by_time

.. autofunction:: ephysalis.legacy.pcclock.pc_fit
```
