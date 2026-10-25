# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for environment and dependency management.

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_decimation.py

# Run a single test by name
uv run pytest tests/test_decimation.py::TestDecimateNpArray::test_decimate_np_array_basic

# Run the CLI tool
uv run ephyscon convert <oe_session_path>
uv run ephyscon convert <oe_session_path> --config config.json --output-folder /output/dir

# Inspect a (SpikeGLX or Open Ephys) session
uv run ephyscon inspect <session_path> --conditions

# Extract and cache sync edges once (slow: reads the AP files), then show trial timing
uv run ephyscon sync <session_path>
uv run ephyscon inspect <session_path> --timing

# Full preprocessing into a work dir (default <session>/ephyscon); sorting needs `uv sync --extra sorting` and a CUDA GPU
uv run ephyscon timing <session_path> --work-dir <dir>
uv run ephyscon lfp <session_path> --work-dir <dir>
uv run ephyscon sort <session_path> --work-dir <dir> --scratch-dir <fast local disk>
uv run ephyscon export <session_path> --work-dir <dir> [--output-format nwb|dh5] [--bids <root>]   # one file from the outputs above

# Type checking (oecon-era modules have known errors; new modules are clean)
uv run mypy src/
```

## Architecture

**ephyscon** (formerly oecon) converts [Open Ephys GUI](https://open-ephys.github.io/gui-docs/) binary recordings into the in-house DAQ-HDF5 (DH5) format used by the cognitive neurophysiology lab.

### Conversion pipeline

The main entry point is `convert_open_ephys_recording_to_dh5()` in `src/ephyscon/convert_open_ephys_to_dh5.py`. It orchestrates the following steps, each controlled by a config dataclass that is `None` to skip that step:

1. **Raw data** (`raw.py`, `RawConfig`) — copies continuous data into DH5 CONT blocks
2. **Events** (`events.py`, `EventPreprocessingConfig`) — extracts TTL triggers (NI-DAQmx / Acquisition Board) and Network Events (VStim), merges and sorts by timestamp, stores as DH5 event triggers
3. **Trial map** (`trialmap.py`, `TrialMapConfig`) — parses `TRIAL_START`/`TRIAL_END` messages from the Open Ephys Message Center (sent by VStim) and writes a structured trial map to DH5
4. **Decimation** (`decimation.py`, `DecimationConfig`) — FIR/IIR anti-alias filter + downsample to produce LFP data; default factor 30×, stored starting at CONT block 2001
5. **Continuous MUA** (`mua.py`, `ContinuousMuaConfig`) — extracts MUA/ESA envelope from raw data; stored starting at CONT block 4001

Each processing function takes a `(config, recording, dh5file)` signature and returns the (potentially updated) config, which is then serialised alongside each output file as `<session>_exp<n>_rec<n>.config.json`.

### Configuration

`OpenEphysConversionConfig` (in `src/ephyscon/config.py`) is the top-level config dataclass — a container of all per-step configs. It serialises/deserialises via `save_config_to_file` / `load_config_from_file` using plain JSON. Enums are not yet fully handled in round-trip deserialization (see TODO in `config.py`).

### Event types

Three event classes in `events.py`:
- `Event` — TTL events with `full_words`, `states`, `sample_numbers`, `timestamps`
- `FullWordEvent` — Network Events after deduplication (no `states` field)
- `Messages` — Open Ephys Message Center text messages

`event_from_eventfolder()` dispatches on `metadata.source_processor` to return the appropriate type.

### Output file naming

Each Open Ephys recording produces one DH5 file and config JSON:
```
<session_name>_exp<experiment_index+1>_rec<recording_index+1>.dh5
<session_name>_exp<experiment_index+1>_rec<recording_index+1>.config.json
```
Both use 1-based indexing in the filenames (exp1, rec1, etc.).

### Tests

Real-data Open Ephys tests are **skipped** unless `tests/data/Test_2026-03-12_16-32-08` and its golden DH5 exist. `tests/test_realdata_spikeglx.py` runs only when `EPHYSCON_TEST_SESSION` points to a SpikeGLX session processed by the MATLAB pipeline. `tests/data/SOMxCD1_172_20250109_crop` is a 5 s real SpikeGLX crop (made with `tools/make_spikeglx_test_dataset.py`) with MATLAB reference output, used by `tests/test_spikeglx_crop.py`. All other tests use mocks or synthetic data and run without external data.

The `pytest-plt` fixture (`plt`) is available in tests for inline plotting during development — tests using it will generate plot files when run with `--plt`.

## Session pipeline (SpikeGLX / multi-run sessions)

This is the new architecture that replaces the lab's MATLAB Neuropixels pipeline (see `docs/roadmap.md`):

- `session.py` — format-independent model: Session → Run → ProbeInfo/streams (`StreamKind`)
- `io/` — `InputFormat` readers (`spikeglx.py`, `openephys.py`) registered under the `ephyscon.input_formats` entry point group. They return lazy spikeinterface recordings, and discovery parses metadata only. Readers detect Neuropixels probes; other probes are defined in `session.toml` (`io/probes.py`, `configure_probes` after loading the config; `Run.get_recording` selects their channels and attaches the probeinterface geometry). Formats with synchronised timestamps (Open Ephys) provide `clock_maps` instead of sync pulses.
- `stimulus/` — `StimulusSystem` plugins (`ephyscon.stimulus_systems`). `stimstage.py` ports the lab's MATLAB `getStimFromIndex.m`, whose conditions are enumerated column-major (MATLAB `ind2sub`), so the field order of `params` matters.
- `sync.py` — sync edges → `ClockMap` per stream onto the reference probe's clock (`RunSync`). `auxsignals.py` — photodiode pulses, wheel speed, `AuxSignals` (role access via `AuxChannelMap`). `timing.py` — `extract_run_timing` (sync, trials of all paradigms, running speed). Formats provide `sync_source` (channel + bit) and `read_aux_*` on `InputFormat`. Sync edges are cached per run by `SyncEdgeCache` (default `<session>/ephyscon/sync/<run>.npz`, filled by `ephyscon sync`), because reading a probe's sync channel reads its whole AP file; code that reads AP data anyway should feed an `EdgeAccumulator` and `SyncEdgeCache.store` the result. Don't name a module `aux.py` (reserved on Windows).
- `sorting.py` — spike sorting per probe over the concatenated runs (sorter from `[sorting]` in `session.toml`, default Kilosort4, via spikeinterface `run_sorter`; SortingAnalyzer on the 300–6000 Hz band-pass filtered copy) (`concatenate_runs`, `sort_probe`), `cut_spikes` back into runs on the reference clock. Steps reuse existing outputs (local copy, sorter folder) so a crashed run can be resumed; inside Kilosort, `kilosort_checkpoints.py` checkpoints the long steps by temporarily replacing `datashift.run`, `spikedetect.run` and `template_matching.extract` (checked against Kilosort 4.1.7). `lfp.py` — per-run LFP at 1 kHz (spikeinterface polyphase `resample`). `nwb.py` — session NWB file from the work-dir outputs; session time 0 is 1 s before the first run, runs are placed by their start times (SpikeGLX `fileCreateTime`, 1 s resolution). `dh5.py` — session DH5 file (LFP CONT blocks per channel with extra INDEX regions for drifting probe clocks, one timestamps-only SPIKE block per unit with `MEAN_WAVEFORM`, TRIALMAP; parameters and unit metrics in `_trials.csv`/`_units.csv` next to it); needs dh5io with SPIKE writing (branch `add-spike-blocks`). `bids.py` — BIDS `ecephys` dataset around the NWB files (sidecars read from the NWB file); DH5 files get a BIDS-like layout with sidecars from `nwb.session_metadata_nwbfile`. `ecephys` is only in BEP032 (unreleased), so validate with the BEP032 schema: `uv run bids-validator-deno <root> --schema https://bids-specification.readthedocs.io/en/bep032/schema.json` (`tests/test_bids.py` does this). The packaged validator lacks permissions for UNC paths (X: share); there, run `deno run -A <venv>/Lib/site-packages/bids_validator_deno/bids-validator.js <root> --schema ...`.
- `matfile.py` — MAT v5/v7.3 reader that keeps struct field order (`MATLAB_fields`)
- `session_config.py` — `SessionConfig` ↔ `session.toml`; `migrate.py` migrates the legacy Excel/txt files. This is unrelated to `config.py`, the Open Ephys → DH5 conversion config.
- CLI: `cli/main.py` (subcommand dispatcher and `convert`), `cli/session_commands.py` (`inspect`, `init`, `sync`, `timing`, `lfp`, `sort`, `export`)

Entry points only change when the package is reinstalled (`uv sync`).

### Local setup notes

The working copy lives on a network share (`X:`). Installing into a `.venv` there fails with file locks, so keep the venv local, e.g. `UV_PROJECT_ENVIRONMENT=C:/Users/<you>/.venvs/ephyscon UV_LINK_MODE=copy`. Git needs `-c safe.directory=*` (or a `safe.directory` entry) on that share.
