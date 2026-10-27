# Historic recording formats

The package `ephysalis.legacy` reads the lab's historic recordings. It will be removed once those
datasets are converted. These pages describe the file formats in enough detail to write another
reader without ephysalis.

| Format | Files | Years | Recordings | Reader |
|---|---|---|---|---|
| {doc}`daq` | `.daq` | 1999–2006 | Fritz, Mini (epidural and intracortical arrays, ADC64/MC-128 cards) | `ephysalis.legacy.daq` |
| {doc}`uff` | `M*.UFF` + `P*.UFF` | 1999 | Eddi (Tallon-Baudry et al. 2004) | `ephysalis.legacy.uff` |
| {doc}`mcd` | `.mcd` | 2010–2011 | Iris (USB-ME256) | `ephysalis.legacy.mcd` |
| {doc}`vstim-td01` | trial records + `.tdr` | 2000–2011 | trials of the DAQ and MC_Rack recordings | `ephysalis.legacy.vstim_td01` |

None of these formats is documented by its authors:
- the DAQ format was reconstructed from Michael Borisov's DAQ → HDF5 converter;
- the UFF format was reverse-engineered in the Tallon-Baudry BIDS conversion;
- the MCD format was reverse-engineered here, with the Multi Channel Systems Neuroshare library
  as reference.

Byte offsets that are given here were checked on real files. Fields marked *unknown* were not
needed and were not identified.

## How the readers map onto ephysalis

All three readers fit the same model as the Open Ephys and SpikeGLX readers:

- **Sessions and runs.** A folder is a session. A run is one recording block:
  - DAQ: a file, or a contiguous part of a paused file;
  - UFF: an M/P file pair;
  - MCD: a chain of files that MC_Rack split one recording into.
- **Streams.** A stream is a group of channels with one sample clock:
  - DAQ: one per board;
  - UFF: `M` and `P`;
  - MCD: one per MC_Rack stream.
- **Probes.** None of the formats stores electrode geometry, so the electrodes are defined in
  `session.toml` as channels of a stream:

  ```toml
  [probes.epidural]
  stream = "ADC64 Rev F"                 # ephysalis inspect lists the streams
  channels = ["AI0", "AI1", "AI2"]
  ```

- **Aux stream.** One stream per run carries the TTL lines and the analog inputs (eye position,
  mains):
  - DAQ: the first board;
  - UFF: `M`;
  - MCD: the digital stream `digi0001`.
- **Trial descriptors.** DAQ and MCD files contain trial and stimulus numbers written by the
  stimulus PC (VStim 1.x). The `vstim_td01` stimulus plugin joins them with VStim's own trial
  file (`.tdr`).
- **Clocks.** Each reader provides clock maps from every stream onto the run's reference clock.
- **Caches.** Indexing a large file on a network share is slow, so the DAQ record index and the
  MCD digital transitions are cached in `EPHYSALIS_CACHE_DIR` (default `%LOCALAPPDATA%/ephysalis`
  or `~/.cache/ephysalis`).

## Isolation

Nothing outside `ephysalis.legacy` imports it, which `tests/test_legacy_isolation.py` checks. The
readers plug in only through the entry points in `pyproject.toml` (`daq`, `uff`, `mcd`,
`vstim_td01`). Removing them means deleting the package, `tests/legacy/` and these four entry
points.

The real recordings may not be published. `tests/legacy/` therefore writes small synthetic files
in each format, and these writers are compact examples of the layouts described here.
