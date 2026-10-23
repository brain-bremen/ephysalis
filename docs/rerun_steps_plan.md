# Plan: Re-run individual conversion steps on existing output files

## Goal

Allow users to re-run a previously-skipped step, or repeat a step with new parameters, without re-converting the entire session. This modifies an existing output file in place.

## Design

### New library functions (`convert_open_ephys_to_dh5.py`)

**Helpers (internal)**
- `_delete_cont_range(h5file, start_id, end_id)` — deletes all `CONT<n>` groups in the given ID range
- `_delete_dataset_if_exists(h5file, name)` — deletes a named dataset if present

**`rerun_steps_on_existing_dh5(recording, session_name, config, on_progress)`**
1. Derives DH5 path from session_name + recording indices (same 1-based naming convention as fresh conversion)
2. Loads existing config JSON (if present) to preserve parameters for steps not being re-run
3. Opens existing DH5 with `DH5File(path, mode="r+")` — no truncation
4. For each step enabled in `config`, deletes the relevant data and re-runs the step
5. Merges new step configs into the loaded config and saves back to the JSON

**`rerun_open_ephys_session(session_path, output_folder, config, on_progress)`**
- Session-level entry point, parallel to `convert_open_ephys_session`
- Exported from `ephyscon.__init__`

### What gets deleted per step

| Step | Deleted before re-running |
|---|---|
| Raw | CONT blocks in raw ID range |
| Events | `EV02` dataset |
| Trial map | nothing (`add_trialmap_to_file` uses `replace=True` internally) |
| LFP | CONT blocks in `[start_block_id, start_block_id + n_channels)` |
| MUA | CONT blocks in `[start_block_id, start_block_id + n_channels)` |

### Config JSON merge

After a partial re-run, the config JSON must reflect the actual state of the file:
- Load the existing config JSON
- Replace only the step-config fields that were re-run
- Save back (overwrite)

This ensures the JSON always matches what is in the output file.

### GUI (`main_window.py`)

- Add a "Re-run Steps" button alongside "Run"
- In re-run mode, unchecked step tabs mean "leave untouched" (not "skip")
- A `_RerunWorker` thread (or a mode flag on `_ConversionWorker`) calls `rerun_open_ephys_session`; the signal interface is identical so progress/log wiring is reused

### CLI (`main.py`)

Add `--rerun` flag:
```
--rerun    Re-run selected steps on existing output file(s).
           Requires --config to specify which steps to re-run.
```

### Output format support

The re-run logic is format-specific. The DH5 implementation uses direct HDF5 group deletion via h5py. When NWB support is added, a parallel `rerun_steps_on_existing_nwb()` function handles NWB-appropriate cleanup. `rerun_open_ephys_session` dispatches based on `config.output_format`.

## Risks

**Partial failure**: if a step crashes after deletion but before writing, the output file is left incomplete. A clear error is logged; no rollback for now.

**MUA depends on decimation params**: even when only MUA is re-run, the current `decimation_config` is passed through — already handled by the existing fallback in the orchestrator.

**HDF5 fragmentation**: deleting HDF5 groups does not reclaim disk space. Users can repack manually with `h5repack` if needed.

## Naming convention (reminder)

Both the DH5 file and the config JSON use **1-based** experiment and recording indices:
```
<session>_exp1_rec1.dh5
<session>_exp1_rec1.config.json
```
