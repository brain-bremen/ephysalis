# Stimulus systems

A stimulus system plugin turns what the stimulus program left behind into a table of trials:
- **when** each trial happened, on the run's reference clock;
- **what** was shown, i.e. the condition and its stimulus parameters;
- **how** the trial went, e.g. the outcome and reaction time.

ephysalis then writes these trials into every output: the NWB trials table, DH5 `TRIALMAP`, and
BIDS `events.tsv`.

Behavioural signals such as running speed, eye position and lever presses do not come from these
plugins. They come from the roles of the aux channels (see [Behaviour signals](#behaviour-signals)).

## How `ephysalis timing` uses a plugin

For each included run, `ephysalis timing` (`ephysalis.timing.extract_run_timing`) does this:

1. **Clocks.** It synchronises the run's streams, which gives a `RunSync`: clock maps from every
   stream onto the reference clock.
2. **Aux signals.** It opens the aux channels with their roles from `[aux_channels]`, which gives
   `AuxSignals`.
3. **Choosing the system.**
   - **Set in `session.toml`:** if the run sets `stimulus_system`, that plugin is used.
   - **Auto-detection:** otherwise every installed plugin is asked `can_handle(run)` in the
     alphabetical order of the entry point names (`stimstage`, `vstim`, `vstim_td01`, …), and the
     first that says yes is used.
   - **None found:** the run has no trials.
4. **Paradigms.** `discover(run)` lists the paradigms of the run in presentation order. A run can
   contain several, e.g. a receptive-field mapping followed by a grating paradigm.
5. **Trials.** `load_trials(run, paradigm, sync, aux)` is called for each paradigm. Each table is
   written to `<work>/trials/<run>_<paradigm>.csv`.
6. **Run start.** `run_start(run, trials)` may return a better start time for the run, from the
   stimulus PC's clock. It is written to `<work>/run_start/<run>.json` (see `ephysalis.runstart`).

When exporting, `export` reads all trial files of the included runs. It adds the columns `run`
and `paradigm`, shifts the times onto the session time axis, and sorts the trials by
`start_time`.

```toml
[[runs]]
name = "experiment1_recording1"
stimulus_system = "vstim"      # optional; auto-detected if not set
```

## The trial table

`load_conditions(paradigm)` returns one row per presented trial, in presentation order, and needs
only the stimulus files. `load_trials` returns the same rows with times added. Trials without a
detected time are dropped, with a warning.

| Column | Required | Meaning | Used by |
|---|---|---|---|
| `trial` | yes | 0-based index within the run and paradigm | NWB, BIDS |
| `condition` | yes | the system's condition (stimulus) id | DH5 `TRIALMAP.StimNo`, NWB, BIDS |
| `start_time` | yes (`load_trials`) | s on the reference clock of `sync` | everywhere |
| `stop_time` | yes (`load_trials`) | s on the reference clock; the start if the end is unknown | everywhere |
| `outcome` | no | VStim `TrialOutcome` code (1 Hit, 5 Early, 6 Late, 7 EyeErr, −1 Undetermined, …) | DH5 `TRIALMAP.Outcome` (default 1 = Hit) |
| `outcome_name` | no | name of the outcome | NWB, BIDS |
| `vstim_trial` | no | trial number of the stimulus program | DH5 `TRIALMAP.TrialNo` if every trial has one, else the position in the session |
| anything else | no | stimulus parameters, reaction times, sources of the times, … | NWB trials columns, BIDS `events.tsv`, DH5 `<file>_trials.csv` |

Other columns become columns of the NWB trials table. Paradigms that lack one of them get NaN
there, or an empty string for text columns. Columns that say where a time came from
(`start_source`, `stop_source`) help later checks: e.g. `trigger` against `message` in the VStim
plugin.

## What a plugin gets

`sync: RunSync`
: Clock maps of all streams. `sync.to_reference_time(stream, samples)` converts fractional
  sample indices of a stream into seconds on the reference clock. The aux stream is
  `ephysalis.sync.AUX`. `sync.to_samples(stream, times)` converts back.

`aux: AuxSignals | None`
: The aux channels by role. It is `None` if the run has no aux stream.
  - `aux.digital("trial_start")`: a boolean array of the line.
  - `aux.analog("photodiode")`: raw counts, and `aux.analog_volts(role)` the same in volts.
  - `aux.has(role)`: whether the role is set.
  - `aux.photodiode_pulses(n_trials)`: the lab's photodiode pulse detection.
  - `aux.sampling_frequency`: the aux stream's rate.
  - `aux.channels`: the role → channel map.

`run: Run`
: The run and its format. It gives the files (`run.path`), the start time, and the format's
  readers:
  - `run.format.read_ttl_events(run, lines)`: TTL edges as `aux_sample`, `line`, `state`;
  - `run.format.read_messages(run)`: text messages, e.g. Open Ephys;
  - format-specific methods, which a plugin should look up with `getattr` so that it still works
    on formats without them (as `vstim_td01` does with `read_trial_descriptors`).

Work in samples of the aux stream and convert to seconds with `sync` at the end, as the
existing plugins do. All streams then line up on the reference clock.

## A minimal plugin

This example plugin reads a CSV of conditions written by the stimulus program and times the
trials by a trigger line:

```python
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from ephysalis.stimulus.base import Paradigm, StimulusSystem
from ephysalis.sync import AUX


class MyStimulusSystem(StimulusSystem):
    name: ClassVar[str] = "mystim"

    def can_handle(self, run):
        return (run.path / "conditions.csv").exists()

    def discover(self, run):
        return [Paradigm(name="main", run_name=run.name, index_in_run=0, system=self.name,
                         path=run.path / "conditions.csv")]

    def load_conditions(self, paradigm):
        table = pd.read_csv(paradigm.path)          # one row per trial: condition, contrast, ...
        table.insert(0, "trial", np.arange(len(table)))
        return table

    def load_trials(self, run, paradigm, sync, aux):
        trials = self.load_conditions(paradigm)
        line = aux.digital("trial_start")
        onsets = np.flatnonzero(~line[:-1] & line[1:]) + 1      # aux samples
        offsets = np.flatnonzero(line[:-1] & ~line[1:]) + 1
        n = min(len(trials), len(onsets), len(offsets))
        trials = trials.iloc[:n].copy()
        trials.insert(1, "start_time", sync.to_reference_time(AUX, onsets[:n].astype(float)))
        trials.insert(2, "stop_time", sync.to_reference_time(AUX, offsets[:n].astype(float)))
        return trials
```

Register it in the `pyproject.toml` of your package. Entry points take effect when the package is
(re)installed.

```toml
[project.entry-points."ephysalis.stimulus_systems"]
mystim = "mypackage.stimulus:MyStimulusSystem"
```

Make `can_handle` cheap and specific. It is called for every run and every installed plugin, and
the first plugin that says yes wins.

## The included plugins

### `vstim`: VStim with Open Ephys

This plugin is for the lab's VStim (C++) with Open Ephys (`ephysalis.stimulus.vstim`).

- **Recording:** VStim writes one TDR file (trial data record) per recording. It also sends
  messages to Open Ephys: `START RECORDING … <file>.tdr`, then `TRIAL_START <n> …` and
  `TRIAL_END <n> …`.
- **Detection:** the TDR file named in the `START RECORDING` message is found next to the session
  or one or two folders above it.
- **Paradigm:** one per run, named after the stimulus definition (`.ini`) in the TDR header.
- **Conditions:** one row per TDR trial: `condition` is VStim's stimulus number, plus
  `vstim_trial`, outcome, time sequence, reaction time, reward and duration.
- **Start:** the rising edge of the `trial_start` line within 1 s before the trial's
  `TRIAL_START` message. The message comes over the network, 0.1–0.5 s late. Without the line,
  the message time is used, with a warning (`start_source`).
- **End:** the start plus the trial duration from the TDR.

### `stimstage`: MATLAB/Psychtoolbox stimulusStage

This plugin is for the lab's MATLAB/Psychtoolbox stimulus code (`ephysalis.stimulus.stimstage`).

- **Detection:** the run folder contains one `<k>_stimulusStage<Name>` folder per paradigm.
- **Parameters:** `params.mat` holds a cell array of parameter structs. Every vector-valued field
  is a factor, and the conditions enumerate the full factorial design of each struct.
- **Order:** `stimulation.mat` holds `stimResults.stimulusOrder`, the condition index per trial.
- **Conditions:** port of `getStimFromIndex.m`. Conditions are numbered over the structs and,
  within a struct, in MATLAB `ind2sub` order (column-major: the first field varies fastest). The
  field order of `params` therefore matters, and ephysalis's MAT reader keeps it. Each row has
  `param_set`, `stimulus_function` and the levels of all fields.
- **Times:** from the screen photodiode (role `photodiode`, with `photodiode_invalid`
  optional), as `fillInSampleIndsInStim_CR.m` does it. In some versions the photodiode square
  covers all stimulus frames, giving start and stop. In others a single-frame pulse marks only the
  onset.
- **Limitation:** runs with several paradigms are not supported, because splitting them by the
  paradigm line is not ported.

### `vstim_td01`: VStim 1.x with DAQ or MC_Rack

This plugin handles VStim 1.x trials with the trial records of the DAQ and MC_Rack files and the
day's `.tdr`. It is described in {doc}`legacy/vstim-td01`. It is also the only plugin with a
`run_start`, the run's start on the VStim PC's clock.

(behaviour-signals)=
## Behaviour signals

These signals are decoded from aux channel roles by `ephysalis.timing`, `ephysalis.auxsignals` and
`ephysalis.annotations`, independent of the stimulus system:

| Role(s) | Output |
|---|---|
| `wheel` (digital) | running speed in cm/s at 1 kHz (port of `get_wheelspeed.m`; 900 pulses per revolution, radius 8 cm): DH5 CONT 3601, NWB `behavior/RunningSpeed` |
| `eye_x`, `eye_y` (analog) | eye position in V at 1 kHz (uncalibrated): DH5 CONT 3602/3603, NWB `behavior/EyeTracking` |
| any digital role | TTL edges in `events/<run>.csv` and DH5 EV02, except `[events] exclude_roles`; pulses (active edge to the opposite one, `[events.lines.<role>]`) in DH5 `/Intervals`, NWB `trigger_pulses`, BIDS `events.tsv` |
| `[[events.markers]]` | named time points from a line's edges or from the trial starts, with an offset: DH5 `/Markers`, NWB `markers` |

A stimulus plugin can use the same roles. For example, `vstim_td01` reads the outcome pulses
(`reward`, `eye_error`, …) when the `.tdr` lacks a trial.

## API

```{eval-rst}
.. autoclass:: ephysalis.stimulus.base.StimulusSystem
   :members:

.. autoclass:: ephysalis.stimulus.base.Paradigm

.. autoclass:: ephysalis.auxsignals.AuxSignals
   :members:

.. autofunction:: ephysalis.stimulus.stimstage.get_stim_from_index
```
