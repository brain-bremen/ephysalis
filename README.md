# ephysalis — Electrophysiology Converter

Like the physalis berry in its papery husk, ephysalis wraps the juicy core of a recording, the
raw signals, in a shell of metadata: probes and channel maps, trials and stimuli, clocks, events,
and the record of how every output was made. The shell protects the data and keeps it good for
longer. One self-describing NWB or DH5 file per session, laid out as a BIDS dataset, can be
shared and archived, and still be understood years later.

Converts extracellular electrophysiology recordings (SpikeGLX/Neuropixels, Open Ephys) into one
[NWB](https://nwb.org) or [DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5) (DH5) file per
session, optionally in a [BIDS](https://bids.neuroimaging.io) dataset. Along the way it extracts
trials, LFP, MUA and CSD, and sorts spikes with
[spikeinterface](https://spikeinterface.readthedocs.io) (Kilosort4 by default).

ephysalis is the successor of [oecon](https://github.com/brain-bremen/oecon) and replaces the lab's
MATLAB Neuropixels preprocessing scripts.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv tool install git+https://github.com/brain-bremen/ephysalis.git
```

## Usage

```bash
ephysalis-gui                                          # GUI: sessions, session.toml, steps
ephysalis inspect <session> --conditions               # runs, probes, streams, paradigms
ephysalis init <session>                               # create session.toml
ephysalis timing <session> --work-dir D:/work/{session}
ephysalis lfp    <session> --work-dir D:/work/{session}
ephysalis sort   <session> --work-dir D:/work/{session} --scratch-dir <fast local disk>
ephysalis export <session> --work-dir D:/work/{session} --output-format nwb|dh5 [--bids <root>]
```

## Documentation

The documentation is at https://brain-bremen.github.io/ephysalis/: steps, `session.toml`, output
layouts, filters, and plugins. It is built from `docs/`; build it locally with:

```bash
uv run --group docs sphinx-build docs docs/_build
```

## Development

```bash
uv sync
uv run pytest
```
