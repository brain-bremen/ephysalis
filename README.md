<img src="docs/_static/ephysalis-logo.svg" alt="ephysalis" width="480">

# ephysalis — Electrophysiology Converter

Like the physalis berry in its papery husk, ephysalis wraps the juicy core of a recording, the
raw signals, in a shell of metadata: probes and channel maps, trials and stimuli, clocks, events,
and the record of how every output was made. The shell protects the data and keeps it good for
longer, so it can be shared and archived, and still be understood years later.

Converts extracellular electrophysiology recordings (SpikeGLX/Neuropixels, Open Ephys) into one
[NWB](https://nwb.org) or [DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5) (DH5) file per
session, optionally in a [BIDS](https://bids.neuroimaging.io) dataset. Along the way it extracts
trials, LFP, MUA and CSD, and sorts spikes with
[spikeinterface](https://spikeinterface.readthedocs.io) (Kilosort4 by default).

ephysalis is the successor of [oecon](https://github.com/brain-bremen/oecon) and replaces the lab's
MATLAB Neuropixels preprocessing scripts.

> **Alpha.** ephysalis is in an early development stage. Its design (the pipeline steps, the
> `session.toml` schema, the work dir and the output layouts) is still changing, without regard
> for compatibility, and it is not ready for use outside the lab's conversion projects.
> Instructions for installing and using it will follow once the design is settled.

## Documentation

https://brain-bremen.github.io/ephysalis/ (for now only the historic recording formats)
