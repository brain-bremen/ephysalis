# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Renamed from oecon to ephyscon.** The package and console scripts are now `ephyscon`, `ephyscon-gui` and `ephyscon-inspect`. The `cli`/`gui` packages moved to `ephyscon.cli`/`ephyscon.gui`.
- **CLI uses subcommands.** The previous `oecon <session>` is now `ephyscon convert <session>`.
- The conversion config field `oecon_version` is now `ephyscon_version`. Old config files still load.

### Added

- **Session model and readers** for SpikeGLX and Open Ephys (`ephyscon.io`), pluggable via the `ephyscon.input_formats` entry point group
- **Stimulus system plugins** (`ephyscon.stimulus`, entry point group `ephyscon.stimulus_systems`), with a first plugin for the lab's MATLAB "stimulusStage" system
- **`session.toml`** per-session configuration, `ephyscon init` (with `--excel` migration of the legacy lab Excel sheets) and `ephyscon inspect`
- MAT v5/v7.3 reader that preserves struct field order (`ephyscon.matfile`)
- **Clock sync** between probes and the aux stream from the sync edges (`ephyscon.sync`), **photodiode trial times** in the stimstage plugin, **running speed** from the wheel encoder (`ephyscon.auxsignals`), combined per run by `ephyscon.timing.extract_run_timing` and shown by `ephyscon inspect --timing`
- **Sync edge cache**: `ephyscon sync` extracts the sync edges of all streams once (probes read in parallel) into `<session>/ephyscon/sync/`; timing reuses them. `EdgeAccumulator` lets later AP passes (sorting) collect the edges on the way
- **Spike sorting** (`ephyscon.sorting`, `ephyscon sort`): Kilosort4 per probe over the concatenated runs (channels selected by `insertion_depth_um`, minus `bad_channels`), from a local binary copy; SortingAnalyzer with waveforms, unit locations and quality metrics; spikes cut back into runs on the reference clock (`spikes/<run>_<probe>.npz`). Kilosort4 and CUDA PyTorch are in the optional `sorting` extra (`uv sync --extra sorting`)
- **Resumable Kilosort4** (`ephyscon.kilosort_checkpoints`): checkpoints after drift correction, spike detection and template matching, so `ephyscon sort` resumes an interrupted run; Kilosort's spike arrays are allocated at the expected size instead of doubling
- **LFP** (`ephyscon.lfp`, `ephyscon lfp`): per run and probe at 1 kHz with an anti-alias filter (polyphase resampling of the NP1 LF band, or of AP for NP2)
- **Session NWB file** (`ephyscon.nwb`, `ephyscon nwb`): one file per session with all included runs and probes on one time axis: electrodes, epochs (runs), trials of all paradigms, Kilosort units with spike times and peak-channel waveforms, LFP per run and probe (gzip), running speed. Subject, lab and institution come from new `session.toml` fields (`[subject]`, `lab`, `institution`)
- `ephyscon timing` writes the trial tables and running speed of every run; all session outputs share one work dir (`sync/`, `trials/`, `running_speed/`, `lfp/`, `sorting/`, `spikes/`)
- Small real SpikeGLX test dataset (`tests/data/SOMxCD1_172_20250109_crop`, 5 s) with MATLAB reference output, and the crop tool `tools/make_spikeglx_test_dataset.py`

- **GUI** (`ephyscon-gui`, formerly `oecon-gui`): new PySide6 application for converting Open Ephys sessions, with groupboxes for Input, Output, Config, Progress, and Log
- **Session inspector** (`ephyscon-inspect`, formerly `oecon-inspect`): new standalone GUI app for browsing Open Ephys sessions without converting; shows recordings, streams, channel names, and event counts per stream
- **Multi-session support**: multiple Open Ephys sessions can be added and converted in one run
- **Config includes session paths and output folder**: saving a config captures everything needed to reproduce a run; loading it re-populates the session list and output folder
- **Progress bars**: per-step/channel and per-session progress bars driven by an `on_progress` callback threaded through the conversion pipeline
- **CLI progress via tqdm**: step and session progress bars in the terminal; log output redirected through `tqdm` to prevent interleaving
- **Persistent settings**: last-used session directory and config file path remembered across restarts
- **Open Ephys session validation**: selected folders are validated as proper sessions; accepts full session root (contains `Record Node *`) or a single experiment folder inside one
- **Config validation**: invalid config values are now caught and reported as human-readable bullet-point lists

### Fixed

- Config JSON filename now uses 1-based experiment/recording indices (`_exp1_rec1.config.json`) consistent with the DH5 output filename

---

## [0.1.5] and earlier

Initial development — conversion pipeline, CLI, pydantic config models, real-data integration tests.
