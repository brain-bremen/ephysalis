# ephyscon — Electrophysiology Converter

Converts extracellular electrophysiology recordings (SpikeGLX/Neuropixels, Open Ephys) into one
[NWB](https://nwb.org) or [DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5) (DH5) file per
session, optionally in a [BIDS](https://bids.neuroimaging.io) dataset. Along the way it extracts
trials, LFP, MUA and CSD, and sorts spikes with
[spikeinterface](https://spikeinterface.readthedocs.io) (Kilosort4 by default).

ephyscon is the successor of [oecon](https://github.com/brain-bremen/oecon) and replaces the lab's
MATLAB Neuropixels preprocessing scripts.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv tool install git+https://github.com/brain-bremen/ephyscon.git
```

## Usage

```bash
ephyscon-gui                                          # GUI: sessions, session.toml, steps
ephyscon inspect <session> --conditions               # runs, probes, streams, paradigms
ephyscon init <session>                               # create session.toml
ephyscon timing <session> --work-dir D:/work/{session}
ephyscon lfp    <session> --work-dir D:/work/{session}
ephyscon sort   <session> --work-dir D:/work/{session} --scratch-dir <fast local disk>
ephyscon export <session> --work-dir D:/work/{session} --output-format nwb|dh5 [--bids <root>]
```

## Documentation

The documentation is at https://brain-bremen.github.io/ephyscon/: steps, `session.toml`, output
layouts, filters, and plugins. It is built from `docs/`; build it locally with:

```bash
uv run --group docs sphinx-build docs docs/_build
```

## Development

```bash
uv sync
uv run pytest
```
