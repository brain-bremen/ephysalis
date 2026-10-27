# ephyscon — Electrophysiology Converter

ephyscon converts extracellular electrophysiology recordings into one
[NWB](https://nwb.org) or [DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5) (DH5) file per
session, optionally inside a [BIDS](https://bids.neuroimaging.io) dataset. On the way it can extract
trials, LFP, MUA and CSD, select channels, remove mains noise, and sort spikes with
[spikeinterface](https://spikeinterface.readthedocs.io) (Kilosort4 by default).

Input formats: SpikeGLX (Neuropixels) and Open Ephys (Binary, NWB and the legacy Open Ephys
format). Readers for the lab's historic recordings are also included (the lab's DAQ program,
DataWave UFF, Multi Channel Systems MC_Rack). They are described in {doc}`legacy/index`.

ephyscon is the successor of [oecon](https://github.com/brain-bremen/oecon) and replaces the lab's
MATLAB Neuropixels preprocessing scripts.

```{toctree}
:maxdepth: 2
:caption: User guide

usage
session-config
preprocessing
output
filters
```

```{toctree}
:maxdepth: 2
:caption: Historic formats

legacy/index
legacy/daq
legacy/uff
legacy/mcd
legacy/vstim-td01
```

```{toctree}
:maxdepth: 2
:caption: Development

extending
roadmap
```
