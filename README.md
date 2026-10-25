# ephyscon — Electrophysiology Converter

Converts extracellular electrophysiology recordings (SpikeGLX/Neuropixels, Open Ephys with
Neuropixels or other probes) into one [NWB](https://nwb.org) or
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
| `timing` | trials of all paradigms (stimulus system plugins: stimstage, VStim), running speed, eye position, events (TTL lines, VStim network events) |
| `lfp` | LFP per run and probe, 1 kHz |
| `mua` | MUA envelope per run and probe, 1 kHz (300 Hz high-pass, rectified) |
| `sort` | spike sorting per probe over all runs |
| `export` | one NWB or DH5 file (optionally with the raw data), optionally into a BIDS dataset |

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
ephyscon export  <session> [<session> ...] --work-dir D:/work/{session} --output-format nwb|dh5 [--raw] [--bids <root>]

# redo the processing of an output file from its embedded processing record
ephyscon reproduce <file.nwb|file.dh5> --work-dir <new dir> [--run]
```

### `session.toml`

One file per session describes which runs are used, the probes, the sorter, the roles of the
aux channels and the events. It replaces the Excel font-colour convention, `Depths.txt`,
`BrokenChannelsProbe<N>_0Index.txt` and `folders2Combine.txt` of the MATLAB pipeline:

```toml
schema_version = 3
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
| CONT 2001 – 4000 | LFP |
| CONT 4001 – 6000 | MUA envelope |
| CONT 6001 | running speed; CONT 6002/6003 eye position x/y |
| SPIKE `10000 · probe index + unit id` | spike times and mean waveform per unit |
| TRIALMAP | trials of all runs |
| EV02 | TTL edges (`+line`/`-line`), VStim network events (`word + 1000`), names as attributes |

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
