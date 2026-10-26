# ephyscon — Electrophysiology Converter

Converts extracellular electrophysiology recordings (SpikeGLX/Neuropixels, Open Ephys in the
Binary, NWB or legacy Open Ephys format, the lab's DAQ files (`.daq`, 1999-2006), with Neuropixels
or other probes) into one [NWB](https://nwb.org) or
[DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5) (DH5) file per session, optionally in a
[BIDS](https://bids.neuroimaging.io) dataset, with spike sorting via
[spikeinterface](https://spikeinterface.readthedocs.io) (Kilosort4 by default).

ephyscon is the successor of [oecon](https://github.com/brain-bremen/oecon) and replaces the lab's
MATLAB Neuropixels preprocessing scripts. See [docs/roadmap.md](docs/roadmap.md) for scope and
status.

Steps (each one optional, outputs in a work dir per session):

| Step | Output |
|---|---|
| `sync` | sync edges of all streams (clock maps between probes and the aux stream) |
| `timing` | trials of all paradigms (stimulus system plugins: stimstage, VStim), running speed, eye position, events (TTL lines, VStim network events); DAQ runs: start on the stimulus PC's clock from the `.tdr` trial stamps |
| `calibrate` | channel gains from a calibration-sine recording (`[calibration]`) |
| `lfp` | LFP per run and probe, 1 kHz; mains removal with the recorded mains (`[line_noise]`) |
| `quality` | good/bad channels per run from the LFP (`[probes.<id>.quality]`); the CSD then uses the good ones |
| `csd` | current source density of 2-D arrays (`[probes.<id>.csd]`, port of Borisov's `dh_csd`) |
| `mua` | MUA envelope per run and probe, 1 kHz (300 Hz high-pass, rectified) |
| `sort` | spike sorting per probe over all runs |
| `export` | one NWB or DH5 file per session (or per run, `--per-run`, on the run's own clock: DAQ files keep the DAQ clock), optionally with the raw data and into a BIDS dataset |

All filters are zero phase (acausal): the LFP at time *t* depends on the raw data up to 10 ms after *t*,
the MUA up to about 13 ms. See [docs/filters.md](docs/filters.md) for every filter and its reach into the
future.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv tool install git+https://github.com/brain-bremen/ephyscon.git
uv tool upgrade ephyscon   # update
```

## GUI

```bash
ephyscon-gui       # converter: sessions, session.toml, steps, outputs
ephyscon-inspect   # Open Ephys session inspector
```

## CLI

Every step takes one or more sessions. With several sessions, per-session paths contain
`{session}` (the session folder name); without `--work-dir` each session uses
`<session>/ephyscon`.

```bash
ephyscon inspect <session> --conditions             # runs, probes, streams, paradigms
ephyscon init <session> [--excel CR_Experiments25.xlsx]   # create session.toml

ephyscon sync    <session> [<session> ...] --work-dir D:/work/{session}
ephyscon timing  <session> [<session> ...] --work-dir D:/work/{session}
ephyscon lfp     <session> [<session> ...] --work-dir D:/work/{session}
ephyscon mua     <session> [<session> ...] --work-dir D:/work/{session}
ephyscon sort    <session> [<session> ...] --work-dir D:/work/{session} --scratch-dir <fast local disk>
ephyscon export  <session> [<session> ...] --work-dir D:/work/{session} --output-format nwb|dh5 [--raw] [--bids <root>] [--per-run]

# redo the processing of an output file from its embedded processing record
ephyscon reproduce <file.nwb|file.dh5> --work-dir <new dir> [--run]
```

### DAQ files (`.daq`)

A folder of `.daq` files is a session, each file (or each contiguous part of a paused file) a
run. The files store no probes, so the electrodes are defined in `session.toml`, as channels
`AI<DAQ channel>` of a board's stream:

```toml
[probes.epidural]
stream = "ADC64 Rev F"                     # board name (ephyscon inspect lists the streams)
channels = ["AI0", "AI1", "AI2"]           # ...
```

Other channels of the first board are the analog inputs (e.g. `[aux_channels.analog] eye_x = 43`),
the trigger lines its TTL lines. Two boards (2006: ADC64 + MC-128) are synchronised from the copies
of trigger lines 1-16 on lines 17-32. Gains are the voltage at the ADC input. The record index of
each file is cached in `%LOCALAPPDATA%/ephyscon` (`EPHYSCON_CACHE_DIR`).

### Calibration, mains removal, CSD (epidural arrays)

The lab's preprocessing of the Fritz/Mini epidural recordings, as `session.toml` settings:

```toml
[calibration]
run = "15hz_after_2005_03_30"      # 100 µV sine on all inputs (or file = "../other/ephyscon/calibration.json")
[line_noise]
enabled = true                     # dh_remove_50hz with the recorded mains (1 kHz LFP)
subtract_reference_mean = true     # as the BIDS conversions (default false: as dh_remove_50hz)
[aux_channels.analog]
mains = 46
[probes.epidural.csd]              # dh_csd V3.1 [RBF]: grid 3 mm, RBF parameter 2.2758, 1 Ohm m
exclude_inputs = ["AI20"]
[probes.epidural.quality]          # channel selection of the BIDS conversions (intracortical: neighbour_radius_um = 2100)
```

Trials of DAQ runs come from the `TD01` records joined with the day's VStim `.tdr` (stimulus
plugin `vstim_td01`), with the trigger roles of the setup, e.g. for 2004-2005:

```toml
[aux_channels.digital]       # bit = trigger line - 1
trial_start = 0              # line 1: interval 0
eye_error = 10               # lines 11-14: outcome pulses
early_error = 11
late_error = 12
reward = 13
lever_pressed = 14
# 2006 setup: trial_start = 0, trial_end = 11 (line 12), reward = 15

[events.lines.lever_pressed]   # pulses (DH5 /Intervals, BIDS events.tsv): active edge to the opposite one
active = "low"
description = "lever pressed"
[[events.markers]]             # DH5 /Markers (trial_start is always written)
name = "derived_sample_onset"
role = "interval2"
offset_ms = 650                # derived, not measured
```

Runs whose `TD01` records all carry the same trial number (the DAQ program did not receive them,
Fritz 2005-04-28) are matched to the day's `.tdr` by time, with the PC clock of the day's other
runs. `timing` also fits each run's start on the stimulus PC's clock from the `.tdr` trial stamps
(`ephyscon.legacy.pcclock`, via the generic `ephyscon.runstart`): it places the runs of a session file and is the BIDS `acq_time`. With
`task = "shapetracking"` in `session.toml` (or per run) the BIDS files get the `task` entity;
`scans.tsv` lists every file with its `acq_time`.

`ephyscon calibrate`, then `lfp` (with mains removal), `quality` and `csd`. On Fritz 2005-03-30 the
LFP after mains removal equals the BIDS dataset sample by sample (with `subtract_reference_mean`),
the channel selection its `channels.tsv`; trials and run starts of 2005-03-30 and 2005-04-28 (with
the frozen runs) equal its trial maps and `scans.tsv`.

### `session.toml`

One file per session describes which runs are used, the probes, the sorter, the roles of the
aux channels and the events. It replaces the Excel font-colour convention, `Depths.txt`,
`BrokenChannelsProbe<N>_0Index.txt` and `folders2Combine.txt` of the MATLAB pipeline:

```toml
schema_version = 5
session_id = "SOMxCD1_172_20250109_Recording1"
subject_id = "SOMxCD1_172"

[[runs]]
name = "3_rfmappingFine_g0"
include = true

[probes.imec0]
area = "V1"
insertion_depth_um = 2000.0
bad_channels = []          # 0-based

[sorting]
sorter = "kilosort4"

[aux_channels.analog]      # analog input per role
photodiode = 2
[aux_channels.digital]     # digital bit per role
wheel = 7

[events]
exclude_roles = ["sync", "wheel"]
```

## DH5 layout

| Block | Content |
|---|---|
| CONT 1 – 1600 | raw AP/wideband data (`export --raw`), one block per channel over all probes |
| CONT 1601 – 2000 | analog inputs of the aux stream (`export --raw`) |
| CONT 2001 – 3600 | LFP (`ChannelStatus` / `ChannelStatusDescription`: channel selection) |
| CONT 3601 | running speed; CONT 3602/3603 eye position x/y |
| CONT 4001 – 5600 | MUA envelope |
| CONT 8001 – 9600 | CSD (int16 scaled as `dh_csd`: max / 32766.99 A/m³ per count) |
| SPIKE `10000 · probe index + unit id` | spike times and mean waveform per unit |
| TRIALMAP | trials of all runs |
| EV02 | TTL edges (`+line`/`-line`), VStim network events (`word + 1000`), names as attributes (`EventCodes`/`EventNames`: `<role>_on` = active edge) |
| /Intervals | `<role>`: pulses of each trigger line with a role; `run_<name>`: the runs |
| /Markers | `trial_start` and the `[[events.markers]]` |

Trial parameters and unit metrics are in `<file>_trials.csv` and `<file>_units.csv` next to it.

## Library

```python
from ephyscon.io.base import open_session
from ephyscon.stimulus.base import detect_stimulus_system

session = open_session("/data/SOMxCD1_172_20250109_Recording1")
run = session.runs[0]
system = detect_stimulus_system(run)
conditions = system.load_conditions(system.discover(run)[0])  # pandas DataFrame, one row per trial
```

## Extending

Input formats and stimulus systems are plugins registered through entry points:

```toml
[project.entry-points."ephyscon.input_formats"]
myformat = "mypackage.reader:MyFormat"          # subclass of ephyscon.io.base.InputFormat

[project.entry-points."ephyscon.stimulus_systems"]
mystim = "mypackage.stimulus:MyStimulusSystem"  # subclass of ephyscon.stimulus.base.StimulusSystem
```

## Development

```bash
uv sync
uv run pytest
```

Tests against a real SpikeGLX session processed by the MATLAB pipeline run when
`EPHYSCON_TEST_SESSION` points to that session folder.
