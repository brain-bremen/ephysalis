# ephyscon — Electrophysiology Converter

Preprocessing of extracellular electrophysiology recordings (Open Ephys, SpikeGLX/Neuropixels)
into [DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5) (DH5) and, soon,
[NWB](https://nwb.org), with spike sorting via [spikeinterface](https://spikeinterface.readthedocs.io)
and Kilosort4.

ephyscon is the successor of [oecon](https://github.com/brain-bremen/oecon) and will also replace
the lab's MATLAB Neuropixels preprocessing scripts. See [docs/roadmap.md](docs/roadmap.md) for
scope and status.

- **Open Ephys → DH5** (`ephyscon convert`): LFP decimation, MUA/ESA envelope, TTL events, VStim
  network events, trial map and raw data copy. This is the former oecon pipeline.
- **SpikeGLX / multi-run sessions** (`ephyscon inspect`, `ephyscon init`): session discovery,
  `session.toml` configuration and stimulus conditions. Sync, sorting and NWB output are in progress.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv tool install git+https://github.com/brain-bremen/ephyscon.git
```

### Update

```bash
uv tool upgrade ephyscon
```

## GUI

```bash
ephyscon-gui       # Open Ephys → DH5 converter
ephyscon-inspect   # Open Ephys session inspector
```

![](docs/oecon-gui.png)

## CLI

```bash
# Open Ephys -> DH5
ephyscon convert <session_path> [<session_path> ...]
ephyscon convert <session_path> --config config.json --output-folder /out
ephyscon convert <session_path> --inspect

# any supported format (SpikeGLX, Open Ephys): runs, probes, streams, paradigms
ephyscon inspect <session_path> --conditions

# create session.toml, optionally migrating the legacy lab Excel sheet
ephyscon init <session_path> --excel CR_Experiments25.xlsx -o session.toml
```

### `session.toml`

One file per session describes which runs are used and what is known about the probes. It
replaces the Excel font-colour convention, `Depths.txt`, `BrokenChannelsProbe<N>_0Index.txt`
and `folders2Combine.txt` of the MATLAB pipeline:

```toml
session_id = "SOMxCD1_172_20250109_Recording1"
subject_id = "SOMxCD1_172"

[[runs]]
name = "3_rfmappingFine_g0"
include = true

[probes.imec0]
area = "V1"
insertion_depth_um = 2000.0
bad_channels = []          # 0-based

[aux_channels.analog]      # NIDQ analog input per role
photodiode = 2
[aux_channels.digital]     # NIDQ digital bit per role
wheel = 7
```

## Library

```python
from ephyscon import convert_open_ephys_session
convert_open_ephys_session("/path/to/session")

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

## Default CONT block ranges (DH5)

| Range       | Purpose |
|---|---|
|    1 – 2000 | Raw neural/analog data |
| 2001 – 3600 | LFP |
| 3601 – 4000 | Downsampled analog |
| 4001 – 5600 | MUA/ESA |
| 6001 – 7600 | High-pass filtered raw (reserved) |

Sorted spikes: SPIKE 1–1000.

## Development

```bash
uv sync
uv run pytest
```

Tests against a real SpikeGLX session processed by the MATLAB pipeline run when
`EPHYSCON_TEST_SESSION` points to that session folder.
