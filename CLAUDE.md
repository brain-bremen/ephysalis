# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for environment and dependency management.

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file / test
uv run pytest tests/test_dh5.py
uv run pytest tests/test_dh5.py::test_lfp_blocks

# Inspect a (SpikeGLX or Open Ephys) session
uv run ephysalis inspect <session_path> --conditions

# Extract and cache sync edges once (slow: reads the AP files), then show trial timing
uv run ephysalis sync <session_path>
uv run ephysalis inspect <session_path> --timing

# Full preprocessing into a work dir (default <session>/ephysalis); sorting needs `uv sync --extra sorting` and a CUDA GPU.
# Every step takes several sessions; per-session paths then need {session}, e.g. --work-dir D:/work/{session}
uv run ephysalis timing <session_path> [<session_path> ...] --work-dir <dir>
uv run ephysalis lfp <session_path> --work-dir <dir>
uv run ephysalis mua <session_path> --work-dir <dir>
uv run ephysalis sort <session_path> --work-dir <dir> --scratch-dir <fast local disk>
uv run ephysalis export <session_path> --work-dir <dir> [--output-format nwb|dh5] [--raw] [--bids <root>]   # one file from the outputs above

# Redo the processing of an output file from its embedded processing record (dry run without --run)
uv run ephysalis reproduce <file.nwb|file.dh5> --work-dir <new dir> [--session <path>] [--run]

# Type checking
uv run mypy src/

# Documentation (Sphinx + MyST, docs/; CI builds it with -W)
uv run --group docs sphinx-build docs docs/_build
```

User documentation lives in `docs/` (the README is only a short entry point). The file formats of the legacy readers are specified in `docs/legacy/<format>.md`; keep them in sync when a reader changes.

## Architecture

**ephysalis** (formerly oecon) converts extracellular electrophysiology sessions (SpikeGLX, Open Ephys) into one NWB or DH5 (the lab's DAQ-HDF5 format) file per session, optionally inside a BIDS dataset. The former Open Ephys → DH5 converter (`ephysalis convert`) was replaced by this pipeline; `tests/data/Test_2026-03-12_16-32-08_exp1_rec1.dh5` (and its `.config.json`) is its output, kept as reference for the raw data and EV02 of the new DH5 writer (`tests/test_raw_mua_events.py`).

### Tests

The Open Ephys tests use `tests/data/Test_2026-03-12_16-32-08` (NI-DAQ aux stream, VStim). `tests/test_openephys_formats.py` runs the Open Ephys reader on one recording in each of the three formats from the open-ephys-python-tools repository (set `OPEN_EPHYS_TOOLS_TEST_DATA` to its `tests/data`; CI clones it at a fixed commit) and on a synthetic Neuropixels Binary recording (`tests/data/openephys_np1_settings.xml` from the probeinterface test data). `tests/test_realdata_spikeglx.py` runs only when `EPHYSALIS_TEST_SESSION` points to a SpikeGLX session processed by the MATLAB pipeline. `tests/data/SOMxCD1_172_20250109_crop` is a 5 s real SpikeGLX crop (made with `tools/make_spikeglx_test_dataset.py`) with MATLAB reference output, used by `tests/test_spikeglx_crop.py` and most pipeline tests. All other tests use synthetic data.

The `pytest-plt` fixture (`plt`) is available in tests for inline plotting during development — tests using it will generate plot files when run with `--plt`.

## Pipeline

This is the new architecture that replaces the lab's MATLAB Neuropixels pipeline (see `docs/roadmap.md`):

- `session.py` — format-independent model: Session → Run → ProbeInfo/streams (`StreamKind`); core code uses only the typed fields (`streams`, `clock_start_s`, `start_time_method`, ...), `Run.extra` is private to the format that wrote it
- `io/` — `InputFormat` readers (`spikeglx.py`, `openephys.py`) registered under the `ephysalis.input_formats` entry point group. They return lazy spikeinterface recordings, and discovery parses metadata only. Readers detect Neuropixels probes; other probes are defined in `session.toml` (`io/probes.py`, `configure_probes` after loading the config; `Run.get_recording` selects their channels and attaches the probeinterface geometry). Formats with synchronised timestamps (Open Ephys, DAQ) provide `clock_maps` instead of sync pulses; `default_aux_channels` gives the aux roles used without `session.toml`. `openephys.py` reads the Binary, NWB and legacy Open Ephys formats with open-ephys-python-tools (format and recording detection, stream metadata, TTL events, messages) and its own lazy `OpenEphysRecordingExtractor` for the samples (the package's legacy loader copies whole channels into memory, so legacy `.continuous` records are memory-mapped directly); Network Events words are rebuilt from the TTL line states in file order (Binary events are loaded unsorted, `sort_events = False`).
- `stimulus/` — `StimulusSystem` plugins (`ephysalis.stimulus_systems`). `vstim.py` reads VStim TDR files (vstim package) and times trials with the trial start trigger line (role `trial_start`) matched to the Open Ephys `TRIAL_START` messages. `stimstage.py` ports the lab's MATLAB `getStimFromIndex.m`, whose conditions are enumerated column-major (MATLAB `ind2sub`), so the field order of `params` matters.
- `legacy/` — readers and plugins for the lab's historic recordings, to be removed once those datasets are converted; nothing outside `legacy/` and `tests/legacy/` may import it (`tests/test_legacy_isolation.py`), it plugs in only through entry points and the generic interfaces. `daq.py` reads the lab's `.daq` files (versions 1.0-3.3; one run per file or per contiguous part, one stream per board, channels `AI<n>`, probes only from `session.toml`; boards synchronised from the copied trigger lines, `board_fit`; record index cached on disk, `cache_dir`). `uff.py` reads DataWave UFF files (Eddi 1999; M/P block pairs, P mapped onto M from the pulses on channel 15). `mcd.py` reads MC_Rack files (Iris 2010-2011; chunks `tag[8] + int64 size`, index from `IDXIDX`, recordings split over several files joined into one run, trigger events with trial/stimulus numbers as `read_trial_descriptors`); it was checked against the 32-bit MCS Neuroshare DLL (`nsMCDLibrary.dll` via ctypes from a 32-bit Python, not part of the repo; note that the DLL mixes up files with the same name in different folders). `vstim_td01.py` gives the trials of DAQ runs: `TD01` records (`DaqFormat.read_trial_descriptors`) + the day's `.tdr` (start = last `trial_start` edge, end from a `trial_end` line, the `.tdr` virtual trigger 11 or the outcome pulse + 335 ms; frozen `TD01` matched by time), and its `run_start` on the VStim PC's clock (`pcclock.py`). The real DAQ/UFF data may not be published: `tests/legacy/` writes synthetic files; the readers were checked against the BIDS conversions' code on real files. Tests of format-independent steps use synthetic Open Ephys sessions (`tests/synthetic.py`), never the legacy formats.
- `sync.py` — sync edges → `ClockMap` per stream onto the reference probe's clock (`RunSync`). `auxsignals.py` — photodiode pulses, wheel speed, `AuxSignals` (role access via `AuxChannelMap`). `timing.py` — `extract_run_timing` (sync, trials of all paradigms, running speed). Formats provide `sync_source` (channel + bit) and `read_aux_*` on `InputFormat`. Sync edges are cached per run by `SyncEdgeCache` (default `<session>/ephysalis/sync/<run>.npz`, filled by `ephysalis sync`), because reading a probe's sync channel reads its whole AP file; code that reads AP data anyway should feed an `EdgeAccumulator` and `SyncEdgeCache.store` the result. Don't name a module `aux.py` (reserved on Windows).
- `sorting.py` — spike sorting per probe over the concatenated runs (sorter from `[sorting]` in `session.toml`, default Kilosort4, via spikeinterface `run_sorter`; SortingAnalyzer on the 300–6000 Hz band-pass filtered copy) (`concatenate_runs`, `sort_probe`), `cut_spikes` back into runs on the reference clock. Steps reuse existing outputs (local copy, sorter folder) so a crashed run can be resumed; inside Kilosort, `kilosort_checkpoints.py` checkpoints the long steps by temporarily replacing `datashift.run`, `spikedetect.run` and `template_matching.extract` (checked against Kilosort 4.1.7). `lfp.py` — per-run LFP at 1 kHz (spikeinterface polyphase `resample`). `nwb.py` — session NWB file from the work-dir outputs; session time 0 is 1 s before the first run, runs are placed by their start times (SpikeGLX `fileCreateTime`, 1 s resolution). `dh5.py` — session DH5 file (raw (`--raw`), LFP and MUA CONT blocks per channel with extra INDEX regions for drifting clocks and for sample periods that are not whole ns (30 kHz), one timestamps-only SPIKE block per unit with `MEAN_WAVEFORM`, TRIALMAP; parameters and unit metrics in `_trials.csv`/`_units.csv` next to it); needs dh5io with SPIKE writing (branch `add-spike-blocks`). `export --per-run` writes one file per run on its own clock (`nwb.time_axis`, `Run.clock_start_s`, e.g. the DAQ clock; BIDS `run-<n>`, n = position among the included runs). `bids.py` — BIDS `ecephys` dataset around the NWB files (sidecars read from the NWB file); DH5 files get a BIDS-like layout with sidecars from `nwb.session_metadata_nwbfile`. `ecephys` is only in BEP032 (unreleased), so validate with the BEP032 schema: `uv run bids-validator-deno <root> --schema https://bids-specification.readthedocs.io/en/bep032/schema.json` (`tests/test_bids.py` does this). The packaged validator lacks permissions for UNC paths (X: share); there, run `deno run -A <venv>/Lib/site-packages/bids_validator_deno/bids-validator.js <root> --schema ...`.
- `matfile.py` — MAT v5/v7.3 reader that keeps struct field order (`MATLAB_fields`)
- `provenance.py` — reproducibility: every step appends its invocation (argv, options, `session.toml`, all installed package versions, ephysalis git commit, input file sizes and SHA-256 of the metadata files) to `<work>/provenance/<step>.json`; `export` embeds all of it as JSON in the output (NWB `scratch/ephysalis_processing`, DH5 attribute `Processing` of `/Operations/nnn_ephysalis_export`); `ephysalis reproduce` replays it. Keep new processing deterministic (fixed seeds, e.g. `sorting.ANALYZER_SEED`); Kilosort on a GPU is not bit-exact.
- `schema.py` — schema versions of the config files (`session.toml`: `schema_version`, `session_config.SCHEMA_VERSION`; conversion JSON: `config_version`, `config.VERSION`). Raise the version and add a migration (`MIGRATIONS[n]`: raw data n → n+1) whenever a config format changes (fields added, removed, renamed, or meanings changed), and list the change in the `session_config` docstring; newer files are rejected.
- `session_config.py` — `SessionConfig` ↔ `session.toml`; `migrate.py` migrates the legacy Excel/txt files.
- `events.py` — TTL edges of the aux stream (`InputFormat.read_ttl_events`, optionally only some `lines`: `[events] lines_without_role = false`) and network events (`read_network_events`, Open Ephys Network Events from VStim) per run on the reference clock (`events/<run>.csv`, written by `timing`); DH5 codes `+line`/`-line` and `word + 1000` as in the old converter. `calibration.py` — `ephysalis calibrate`: gains from a calibration-sine run (`fit_sine`), `<work>/calibration.json`, applied as `ProbeInfo.gain_factors` in every step (`_configure` in the CLI). `linenoise.py` — port of Borisov's MATLAB `dh_remove_50hz` (recorded mains, `[line_noise]`), in place on the 1 kHz LFP by `ephysalis lfp`. `csd.py` — port of Borisov's C++ `dh_csd` V3.1 [RBF] as one operator matrix (`ephysalis csd`, `[probes.<id>.csd]`); DH5 CONT 8001… with `dh_csd`'s int16 scaling. `tests/test_preprocessing_lab.py` checks both against literal ports of the MATLAB/C++ code. `quality.py` — `ephysalis quality`: channel selection per run on the LFP (port of the BIDS conversions' `channel_quality`), `<work>/quality/<run>_<probe>.json`; the CSD takes the good channels, so CSD channels can differ between runs (DH5 blocks then have the regions of their runs only). `annotations.py` — pulses of the trigger lines (active level from `[events.lines]`) and markers (`[[events.markers]]`) for DH5 `/Intervals`/`/Markers`, NWB `trigger_pulses`/`markers`, BIDS `events.tsv`. `runstart.py` — better run start times from the stimulus system (`StimulusSystem.run_start`), written by `timing` to `<work>/run_start/`, applied to `Run.start_time` in `_configure` (`Run.recorded_start_time` keeps the recording's; BIDS `acq_time`, `scans.tsv`). `mua.py` — MUA envelope like `lfp.py` (300 Hz Butterworth high-pass forward-backward, rectified, resampled to 1 kHz). All filters are acausal (zero phase); `docs/filters.md` lists each one with its reach into the future (checked by `tests/test_filters.py`), and the descriptions written to NWB/DH5/BIDS come from `lfp.LFP_FILTERING` / `mua.MUA_FILTERING`; update all three when a filter changes.
- CLI: `cli/main.py` (subcommand dispatcher), `cli/session_commands.py` (`inspect`, `init`, `sync`, `timing`, `calibrate`, `lfp`, `quality`, `csd`, `mua`, `sort`, `export`, `reproduce`); `per_session` runs a step for several sessions (placeholder `{session}` in per-session paths, failures don't stop the others).

GUI (`gui/`, PySide6): `main_window.py` shows the converter page (`session_page.py`): a session list with a work dir template, the `session.toml` of the selected session ("Apply to sessions…" copies sorting, aux channels, events etc. to the others), and the steps for all sessions. Each step runs as `python -m ephysalis.cli.main <step>` in a `QProcess` (`pipeline.step_commands`), so sorting cannot take the GUI down; a failing session is skipped and the rest continue; `config_widget.ConfigStepWidget` generates forms from pydantic models (used for `session.toml`). GUI tests run offscreen (`tests/test_gui_session.py`).

Entry points only change when the package is reinstalled (`uv sync`).

### Local setup notes

The working copy lives on a network share (`X:`). Installing into a `.venv` there fails with file locks, so keep the venv local, e.g. `UV_PROJECT_ENVIRONMENT=C:/Users/<you>/.venvs/ephysalis UV_LINK_MODE=copy`. Git needs `-c safe.directory=*` (or a `safe.directory` entry) on that share.
