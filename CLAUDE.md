# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for environment and dependency management.

```bash
uv sync                                    # install (entry points only change on reinstall)
uv run pytest                              # all tests; single: uv run pytest tests/test_dh5.py::test_lfp_blocks
uv run mypy src/                           # type checking
uv run --group docs sphinx-build docs docs/_build   # documentation (CI builds it with -W)

# One session (every step also takes several; per-session paths then need {session}, e.g. --work-dir D:/work/{session})
uv run ephysalis inspect <session> --conditions
uv run ephysalis align-clocks <session> --work-dir <dir>      # sync edges (cached; reads the AP files) -> clocks/
uv run ephysalis extract-trials|extract-events|extract-behavior <session> --work-dir <dir>
uv run ephysalis calibrate|lfp|quality|csd|mua <session> --work-dir <dir>
uv run ephysalis sort <session> --work-dir <dir> --scratch-dir <fast local disk>   # needs `uv sync --extra sorting`
uv run ephysalis export <session> --work-dir <dir> [--output-format nwb|dh5] [--per-run] [--raw] [--bids <root>]
uv run ephysalis reproduce <file.nwb|file.dh5> --work-dir <new dir> [--session <path>] [--run]

# A dataset (dataset.toml: defaults, per-session overrides, steps and their options)
uv run ephysalis run dataset.toml [--recording-sessions a,b] [--steps lfp,export] [--parallel N] [--dry-run]
uv run ephysalis status dataset.toml
uv run ephysalis pause dataset.toml        # running steps finish, then the runners stop; `run` resumes
```

The documentation (`docs/`) has only the overview and the roadmap while the design is in its alpha stage (no
user guide yet). Lab-specific readers and metadata plugins live in their own repositories:
[ephysalis-bremen](https://github.com/brain-bremen/ephysalis-bremen) (DAQ with calibration, MC_Rack, UFF; VStim,
VStim 1.x trials; the historic formats' documentation) and
[ephysalis-freiburg](https://github.com/brain-bremen/ephysalis-freiburg) (`psychtoolbox_neuropixels_freiburg`).
Nothing in the core may know about them.

## Architecture

**ephysalis** (formerly oecon) converts extracellular electrophysiology sessions into one NWB or DH5 (the lab's
DAQ-HDF5 format) file per session or run, optionally inside a BIDS dataset.

- `plugins.py` — the plugin interface (`PLUGIN_API_VERSION`): runtime-checkable Protocols. Input formats
  (`ephysalis.input_formats` entry points) have a required core (`can_read_recording_session_folder`, `open_recording_session`,
  `get_stream_recording`, `default_auxiliary_channel_roles`) and optional capabilities (`ReadsDigitalLineEdges`,
  `ReadsAuxiliaryAnalogInputs`, `LocatesSyncPulseChannels`, `HasSynchronisedTimestamps`, `ReadsEventWords`,
  `ReadsTextMessages`, `ReadsTrialDescriptorEvents`, `ListsStreamChannelIds`, `ComputesChannelGainCalibration`).
  Metadata plugins (`ephysalis.metadata_plugins`) have `metadata_plugin_name`/`plugin_api_version` (input formats:
  `input_format_name`; both must equal the entry point name) and `ExtractsTrialTables`,
  `EstimatesRunStartTime`, `AddsSessionConfigFromMetadataFiles`, `NamesEventWords`. Plugins are usually modules of
  functions; loading checks them (`plugin_problems`) and skips misfits with a warning; `require` before using a
  capability. `registry.py` gives typed entry point targets, so pyright checks a module against the Protocols.
  Use long, explicit names for everything (functions, fields, variables): `experimental_protocol_name`, `metadata_plugin_name`,
  `_find_experimental_protocols_in_run`, `recording_session`, never a bare `name`, `system`, `path`, `session`
  or `_discover`. Never write into or delete in the recording session folders (source data).
- `recording_session/` — `model.py` (format-independent RecordingSession → Run → ProbeInfo/streams, `StreamKind`;
  `Run.extra` is private to the format), `config.py` (`RecordingSessionConfig` ↔ `session.toml`, `SCHEMA_VERSION` 8 + `MIGRATIONS`),
  `schema.py` (`migrate`; raise the version and add a migration whenever a config format changes), `probes.py`
  (probes defined in `session.toml`, geometry files relative to the config).
- `formats/input/` — `spikeglx.py`, `openephys.py` (Binary, NWB, legacy Open Ephys via open-ephys-python-tools, own
  lazy extractor for `.continuous`), `base.py` (`open_recording_session`). `formats/output/` — `nwb.py`, `dh5.py` (CONT
  ranges from `dhspec`, SPIKE blocks need dh5io with SPIKE writing), `bids.py` (BEP032 `ecephys`; validate with
  `bids-validator-deno --schema https://bids-specification.readthedocs.io/en/bep032/schema.json`). `matfile.py`.
- `metadata/` — `base.py` (`run_trial_tables`: the run's `metadata_plugin` or the first plugin that recognises it;
  `ExperimentalProtocol`: one experimental protocol of a run, with or without stimulation, one trial table), `runstart.py` (better run start times, `<work>/run_start/`, applied in `_configure`).
- `timeline/` — `clocks.py` (sync edges → `ClockMap` per stream on the reference clock, `RunSync`, `SyncEdgeCache`
  in `<work>/sync`), `steps.py` (the align-clocks and extract steps), `events.py` (digital line edges and event
  words per run, `events/<run>.csv`; DH5 codes made at export), `annotations.py`, `auxsignals.py` (photodiode,
  wheel, eye; role access via `AuxChannelMap`). Don't name a module `aux.py` (reserved on Windows).
- `processing/` — `lfp.py`, `mua.py`, `csd.py` (port of Borisov's `dh_csd`), `linenoise.py` (port of
  `dh_remove_50hz`), `quality.py` (channel selection). All filters are acausal; `docs/filters.md` and
  `lfp.LFP_FILTERING` / `mua.MUA_FILTERING` describe them — update together.
- `sorting/` — `spikes.py` (sorting per probe over the concatenated runs, resumable from its outputs, `cut_spikes`),
  `kilosort_checkpoints.py`.
- `filesystem.py` — `remove_folder`: the only way to delete folders (NFS servers whose directory listing skips
  entries during deletion make `shutil.rmtree` fail); call it before spikeinterface overwrites a folder.
- `provenance.py` — every step appends its invocation to `<work>/provenance/<step>.json`; `export` embeds all of it
  in the output; `ephysalis reproduce` replays it. Keep processing deterministic.
- `dataset/` — `config.py` (`dataset.toml`: `[[recording_sessions]]`, defaults < recording session file < inline
  overrides, `work_dir` required and outside the recording session folders, `STEP_INPUTS` = the step
  graph, step options checked against the step parsers), `runner.py` (`ephysalis run`: steps as subprocesses,
  stamps of inputs and code in `<work>/runner/<step>.json`, logs, locks, pause file and `status.tsv` in
  `state_dir`).
- `cli/` — `main.py`, `recording_session_commands.py` (the steps; `per_recording_session` for several), `dataset_commands.py`
  (`run`, `status`, `pause`). GUI (`gui/`, PySide6) runs each step as `python -m ephysalis.cli.main <step>`.

### Tests

`tests/data/SOMxCD1_172_20250109_crop` is a 5 s real SpikeGLX crop with MATLAB reference output (most pipeline
tests); its trials come from the test metadata plugin `tests/support/trial_table_plugin.py` (registered by a
conftest fixture; subprocesses do not see it). `tests/data/Test_2026-03-12_16-32-08` is an Open Ephys session
(NI-DAQ aux, VStim); `tests/test_openephys_formats.py` also uses open-ephys-python-tools' test data
(`OPEN_EPHYS_TOOLS_TEST_DATA`). `tests/synthetic.py` writes synthetic Open Ephys sessions. The plugin
repositories run their tests against this repository's `tests/data` (`EPHYSALIS_TEST_DATA`).

### Local setup notes

Working copies on a network share: keep the venv on a local disk (`UV_PROJECT_ENVIRONMENT`, `UV_LINK_MODE=copy`),
git needs `safe.directory`. A stale `src/ephysalis.egg-info` (from an old install) shadows the entry points of the
current code: delete it.
