# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Renamed from oecon to ephyscon.** The package and console scripts are now `ephyscon`, `ephyscon-gui` and `ephyscon-inspect`. The `cli`/`gui` packages moved to `ephyscon.cli`/`ephyscon.gui`.
- **One converter for all inputs and outputs.** The Open Ephys → DH5 converter of oecon (`ephyscon convert`, its JSON config and GUI tab) is replaced by the session pipeline, which reads SpikeGLX and Open Ephys and writes NWB or DH5 (optionally in a BIDS dataset). Its features are now pipeline parts: raw data (`export --raw`; DH5 CONT 1…, analog aux inputs at CONT 1601…, before CONT 1…), MUA (`ephyscon mua`, CONT 4001…), TTL and VStim network events (`timing` → `events/<run>.csv`, DH5 EV02 with the same codes, NWB `events` table), trials (VStim plugin)
- **Open Ephys reader on open-ephys-python-tools**: reads all three formats of the Open Ephys GUI (Binary, NWB, legacy Open Ephys `.continuous`), instead of only Binary through neo. Stream names, gains, events and messages of Binary recordings are unchanged (checked on the VStim test session); Neuropixels probes still get their geometry from `settings.xml`
- **DH5 block ids follow the CONT ranges shared by the lab's converters** (`dhspec.cont.CONT_BLOCK_RANGES` in dh5io, also used by the BIDS conversions): LFP 2001–3600, MUA 4001–5600 (at most 1600 probe channels), running speed and eye position x/y moved from CONT 6001–6003 (the range of high-pass filtered raw data) to CONT 3601–3603 (downsampled analog)
- **Several sessions per command** (`ephyscon timing S1 S2 --work-dir D:/work/{session}`) and in the GUI (session list, "Apply to sessions…", steps for all sessions; a failing session does not stop the others)

### Added

- **Session model and readers** for SpikeGLX and Open Ephys (`ephyscon.io`), pluggable via the `ephyscon.input_formats` entry point group
- **Stimulus system plugins** (`ephyscon.stimulus`, entry point group `ephyscon.stimulus_systems`), with a first plugin for the lab's MATLAB "stimulusStage" system
- **`session.toml`** per-session configuration, `ephyscon init` (with `--excel` migration of the legacy lab Excel sheets) and `ephyscon inspect`
- **Reproducible processing:** every step records its command line, options, `session.toml`, the complete Python environment (and ephyscon git commit) and a fingerprint of the input files in `<work>/provenance/`; `ephyscon export` embeds this processing record in the NWB (`scratch/ephyscon_processing`) and DH5 (`/Operations/nnn_ephyscon_export`, attribute `Processing`) file; `ephyscon reproduce <file> --work-dir <dir> [--run]` replays it and warns about changed software or inputs (`ephyscon.provenance`)
- **Schema versions with migrations** for `session.toml` (`ephyscon.schema`); `session.toml` is now schema version 3 (`[events]`; older files are upgraded on loading)
- GUI session page: open and save the session config anywhere (Open…, Save as…); it remembers the config file of each session
- MAT v5/v7.3 reader that preserves struct field order (`ephyscon.matfile`)
- **Clock sync** between probes and the aux stream from the sync edges (`ephyscon.sync`), **photodiode trial times** in the stimstage plugin, **running speed** from the wheel encoder (`ephyscon.auxsignals`), combined per run by `ephyscon.timing.extract_run_timing` and shown by `ephyscon inspect --timing`
- **Sync edge cache**: `ephyscon sync` extracts the sync edges of all streams once (probes read in parallel) into `<session>/ephyscon/sync/`; timing reuses them. `EdgeAccumulator` lets later AP passes (sorting) collect the edges on the way
- **Spike sorting** (`ephyscon.sorting`, `ephyscon sort`): Kilosort4 per probe over the concatenated runs (channels selected by `insertion_depth_um`, minus `bad_channels`), from a local binary copy; SortingAnalyzer with waveforms, unit locations and quality metrics; spikes cut back into runs on the reference clock (`spikes/<run>_<probe>.npz`). Kilosort4 and CUDA PyTorch are in the optional `sorting` extra (`uv sync --extra sorting`)
- **GUI: session pipeline page** (`ephyscon-gui`, first tab; the Open Ephys → DH5 conversion is the second): pick a SpikeGLX/Open Ephys session and work dir, see runs, probes and paradigms, edit all of `session.toml` (general, subject, runs, probes incl. non-Neuropixels definitions, sorter, aux channel roles), run sync/timing/LFP/sorting/export as separate `ephyscon` processes with a log, and see the outputs in the work dir. The config form generator handles nested models, integer lists and parameter dicts
- `ephyscon sync` takes `--work-dir` and `--config` like the other steps
- **VStim stimulus plugin** (`ephyscon.stimulus.vstim`): trials from the TDR file (trial type as `condition`, outcome, time sequence, reaction and reward time), timed by VStim's trial start trigger line on the aux stream and matched to the `TRIAL_START` messages in Open Ephys (message times as fallback). Needs vstim 0.3.0
- **Eye position** (`eye_x`/`eye_y` analog roles; Open Ephys/VStim default AI0/AI2): 1 ms bin means on the reference clock (`eye/<run>.npy`, from `ephyscon timing`), in NWB as `behavior/EyeTracking` and in DH5 as CONT6002/6003
- **Open Ephys aux signals**: analog inputs (`AI<n>`/`ADC<n+1>`), TTL lines as digital bits, Message Center messages (`InputFormat.read_messages`); behaviour-only sessions (no probes) can be exported
- **Non-Neuropixels probes**: `[probes.<id>]` with `stream`, `channels` and `geometry` (probeinterface library `manufacturer`/`model` or a probeinterface `file`, optional headstage `wiring`) defines probes on streams the reader does not detect, e.g. on the Open Ephys acquisition board (`ephyscon.io.probes`)
- **Open Ephys sessions in the session pipeline**: run start times from `sync_messages.txt` (ms, UTC) or `settings.xml`, and clock maps from Open Ephys' own synchronised `timestamps.npy` instead of sync pulses (`InputFormat.clock_maps`)
- **Choice of spike sorter**: `[sorting]` in `session.toml` (`sorter`, `params`; per probe under `[probes.<id>.sorting]`) or `ephyscon sort --sorter`; any sorter of `spikeinterface.sorters`. Kilosort4 stays the default, and its channel neighbourhoods are limited to the channel count for small probes. The sorter and its parameters are recorded in `sorting/<probe>/segments.json`. Waveforms, templates and metrics are now computed on the 300–6000 Hz band-pass filtered data, which wideband streams (Neuropixels 2.0, most other probes) need
- **Resumable Kilosort4** (`ephyscon.kilosort_checkpoints`): checkpoints after drift correction, spike detection and template matching, so `ephyscon sort` resumes an interrupted run; Kilosort's spike arrays are allocated at the expected size instead of doubling
- **LFP** (`ephyscon.lfp`, `ephyscon lfp`): per run and probe at 1 kHz with an anti-alias filter (polyphase resampling of the NP1 LF band, or of AP for NP2)
- **Session NWB file** (`ephyscon.nwb`, `ephyscon nwb`): one file per session with all included runs and probes on one time axis: electrodes, epochs (runs), trials of all paradigms, Kilosort units with spike times and peak-channel waveforms, LFP per run and probe (gzip), running speed. Subject, lab and institution come from new `session.toml` fields (`[subject]`, `lab`, `institution`)
- **Session DH5 file** (`ephyscon.dh5`): LFP (one CONT block per channel, ids from 2001, extra INDEX regions so drifting probe clocks stay within 0.1 sample), running speed (CONT6001), one SPIKE block per unit (timestamps only, mean waveform as `MEAN_WAVEFORM`), TRIALMAP; stimulus parameters and unit metrics in `<name>_trials.csv`/`<name>_units.csv`. Needs dh5io with SPIKE block writing
- **`ephyscon export`** replaces `nwb` and `bids`: `--output-format nwb|dh5`, `--bids <root>` (BIDS for NWB, BIDS-like for DH5, which BIDS does not allow as data format), `--existing` to copy a written file
- **BIDS layout** (`ephyscon.bids`): puts the session NWB file into a BIDS microelectrode electrophysiology (`ecephys`) dataset with `_ecephys.json`, `_probes.tsv`, `_electrodes.tsv`, `_channels.tsv` and `_events.tsv` (runs and trials) made from the NWB file, plus `participants.tsv` and `sessions.tsv`. Follows BEP032 (`ecephys` is not in a released BIDS version yet) and passes the BIDS validator with the BEP032 schema. `--nwb` copies an existing NWB file instead of writing a new one
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

- DH5 CONT blocks at 30 kHz got one INDEX region for the whole recording although 33333 ns is not 1/30 kHz (about 36 ms error per hour); new regions are added before the error reaches 0.1 sample
- SortingAnalyzer waveforms and noise levels used unseeded random spikes and chunks; now seeded (`sorting.ANALYZER_SEED`)

---

## [0.1.5] and earlier

Initial development — conversion pipeline, CLI, pydantic config models, real-data integration tests.
