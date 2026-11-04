# ephysalis — Electrophysiology Converter

Like the physalis berry in its papery husk, ephysalis wraps the juicy core of a recording, the
raw signals, in a shell of metadata: probes and channel maps, trials and stimuli, clocks, events,
and the record of how every output was made. The shell protects the data and keeps it good for
longer, so it can be shared and archived, and still be understood years later.

ephysalis converts extracellular electrophysiology recordings into one
[NWB](https://nwb.org) or [DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5) (DH5) file per
session, optionally inside a [BIDS](https://bids.neuroimaging.io) dataset. On the way it can extract
trials, LFP, MUA and CSD, select channels, remove mains noise, and sort spikes with
[spikeinterface](https://spikeinterface.readthedocs.io) (Kilosort4 by default).

Input formats: SpikeGLX (Neuropixels) and Open Ephys (Binary, NWB and the legacy Open Ephys
format). Further input formats and the metadata plugins (trials, run start times and config
suggestions from the files of a lab setup) are separate packages that register through entry
points (`ephysalis.plugins`):
[ephysalis-bremen](https://github.com/brain-bremen/ephysalis-bremen) (the lab's DAQ files, MC_Rack,
DataWave UFF, VStim) and [ephysalis-freiburg](https://github.com/brain-bremen/ephysalis-freiburg)
(the Freiburg Neuropixels setup).

ephysalis is the successor of [oecon](https://github.com/brain-bremen/oecon) and replaces the lab's
MATLAB Neuropixels preprocessing scripts.

```{note}
**Alpha.** ephysalis is in an early development stage. Its design (the pipeline steps, the
`session.toml` schema, the work dir and the output layouts) is still changing, without regard for
compatibility, and it is not ready for use outside the lab's conversion projects. Instructions for
installing and using it will follow once the design is settled.
```

```{toctree}
:maxdepth: 2
:caption: Reference

configuration
```

```{toctree}
:maxdepth: 2
:caption: Development

roadmap
```
