# NWB Implementation Plan (Phase 3 & 4)

## Current Status (Phase 1 & 2 Complete)

The FileWriter abstraction layer is fully implemented and all processing modules
have been migrated. DH5 output works identically to before.

### What exists now

- `src/ephyscon/file_writer.py` — `FileWriter` ABC, `DH5Writer`, `create_file_writer()` factory
- `src/ephyscon/config.py` — `OutputFormat`, `DH5OutputOptions`, `NWBOutputOptions`, `OpenEphysConversionConfig`
- All processing modules (`raw.py`, `events.py`, `trialmap.py`, `decimation.py`, `mua.py`) use `FileWriter`
- `tests/test_file_writer.py` — 12 unit tests for `DH5Writer`

---

## Phase 3: Add NWB Support

### Step 1 — Add pynwb dependency

In `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "pynwb>=2.0.0",
    "hdmf>=3.0.0",
]
```

Run `uv sync` after adding.

### Step 2 — Implement `NWBWriter` in `src/ephyscon/file_writer.py`

```python
class NWBWriter(FileWriter):
    def __init__(self, nwbfile, io_object, options=None):
        self._nwbfile = nwbfile
        self._io = io_object
        self._options = options or NWBOutputOptions()
        self._electrode_table_created = False

    @property
    def filename(self) -> str:
        return self._io.path

    def write_continuous_data(self, data, channel_info, sample_rate_hz,
                              start_time_ns, channel_name, group_id, calibration):
        from pynwb.ecephys import ElectricalSeries, LFP
        import numpy as np

        # Map group_id ranges to NWB processing modules:
        # 1–1600   (RAW) → acquisition/ElectricalSeries
        # 2001–4000 (LFP) → processing["lfp"]/LFP/ElectricalSeries
        # 4001–6000 (MUA) → processing["mua"]/ElectricalSeries

        # Ensure electrode table exists
        if not self._electrode_table_created:
            self._nwbfile.create_electrode_column(name="label", description="channel label")
            self._electrode_table_created = True

        if group_id <= 1600:
            # RAW → acquisition
            es = ElectricalSeries(
                name=f"ElectricalSeries_{group_id}",
                data=data.squeeze(),
                timestamps=np.array([start_time_ns / 1e9]),
                description=channel_name,
                conversion=float(calibration) if not hasattr(calibration, '__len__') else float(calibration[0]),
            )
            self._nwbfile.add_acquisition(es)

        elif group_id <= 4000:
            # LFP → processing["lfp"]
            if "lfp" not in self._nwbfile.processing:
                self._nwbfile.create_processing_module("lfp", "Local field potential data")
            lfp_module = self._nwbfile.processing["lfp"]
            if "LFP" not in lfp_module.data_interfaces:
                lfp_module.add(LFP(name="LFP"))
            es = ElectricalSeries(
                name=f"ElectricalSeries_{group_id}",
                data=data.squeeze(),
                timestamps=np.array([start_time_ns / 1e9]),
                description=channel_name,
            )
            lfp_module["LFP"].add_electrical_series(es)

        else:
            # MUA → processing["mua"]
            if "mua" not in self._nwbfile.processing:
                self._nwbfile.create_processing_module("mua", "Multi-unit activity envelope")
            mua_module = self._nwbfile.processing["mua"]
            es = ElectricalSeries(
                name=f"ElectricalSeries_{group_id}",
                data=data.squeeze(),
                timestamps=np.array([start_time_ns / 1e9]),
                description=channel_name,
            )
            mua_module.add(es)

    def write_event_triggers(self, timestamps_ns, event_codes, event_code_names=None):
        import numpy as np
        # Store as annotations TimeSeries in acquisition
        from pynwb.misc import AnnotationSeries
        annotations = [str(c) for c in event_codes]
        ts = AnnotationSeries(
            name="EventTriggers",
            data=annotations,
            timestamps=timestamps_ns / 1e9,
            description="TTL and network event triggers",
        )
        self._nwbfile.add_acquisition(ts)

        # Store event code names as metadata attribute if provided
        if event_code_names:
            ts.fields["event_code_names"] = str(event_code_names)

    def write_trialmap(self, trial_data, outcome_mappings, brainbox_mappings=None):
        # Map to NWB trials table
        self._nwbfile.add_trial_column(name="outcome", description="Trial outcome code")
        self._nwbfile.add_trial_column(name="stim_no", description="Stimulus number")

        for trial in trial_data:
            self._nwbfile.add_trial(
                start_time=float(trial["StartTime"]) / 1e9,
                stop_time=float(trial["EndTime"]) / 1e9,
                outcome=int(trial["Outcome"]),
                stim_no=int(trial["StimNo"]),
            )

    def add_operation(self, operation_name, tool_version, metadata=None):
        import json
        from datetime import datetime

        if "processing_history" not in self._nwbfile.processing:
            self._nwbfile.create_processing_module(
                name="processing_history",
                description="Processing operations history for reproducibility",
            )

        history_module = self._nwbfile.processing["processing_history"]
        op_data = {
            "name": operation_name,
            "tool": tool_version,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            op_data["metadata"] = {k: int(v) if hasattr(v, 'item') else v
                                   for k, v in metadata.items()}

        existing = json.loads(history_module.description.split("OPERATIONS:", 1)[-1]
                              if "OPERATIONS:" in history_module.description else "[]")
        existing.append(op_data)
        history_module.description = "Processing operations history for reproducibility\nOPERATIONS:" + json.dumps(existing)

    def close(self):
        self._io.write(self._nwbfile)
        self._io.close()
```

### Step 3 — Add NWB case to `create_file_writer()`

```python
case "nwb":
    from pynwb import NWBFile, NWBHDF5IO
    from datetime import datetime
    from ephyscon.config import NWBOutputOptions

    if nwb_options is None:
        nwb_options = NWBOutputOptions()

    nwbfile = NWBFile(
        session_description=nwb_options.session_description,
        identifier=str(Path(filename).stem),
        session_start_time=datetime.now().astimezone(),
        experimenter=nwb_options.experimenter,
        institution=nwb_options.institution,
        lab=nwb_options.lab,
    )

    io = NWBHDF5IO(str(filename), mode="w")
    return NWBWriter(nwbfile, io, options=nwb_options)
```

### DH5 ↔ NWB Mappings

| DH5 Structure     | NWB Equivalent                           | Notes                        |
|-------------------|------------------------------------------|------------------------------|
| CONT 1–1600 (RAW) | `acquisition/ElectricalSeries`           | Raw acquisition data         |
| CONT 2001–4000 (LFP) | `processing["lfp"]/LFP/ElectricalSeries` | Decimated signals         |
| CONT 4001–6000 (MUA) | `processing["mua"]/ElectricalSeries` | MUA envelope             |
| `/EV02` dataset   | `acquisition/AnnotationSeries`           | No exact NWB equivalent      |
| `/TRIALMAP`       | `trials` table                           | NWB native structure         |
| `/Operations`     | `processing["processing_history"]`       | Custom module, JSON-encoded  |

---

## Phase 4: Integration Tests

Create `tests/test_multiformat.py`:

```python
# Tests to write:
# - test_dh5_output_unchanged: convert real data, compare to golden file
# - test_nwb_output_valid: convert to NWB, validate with pynwb
# - test_data_equivalence: same recording to both formats, compare arrays
```

Key assertions:
- NWB file opens without error via `NWBHDF5IO`
- `nwbfile.acquisition` contains expected ElectricalSeries
- `nwbfile.trials` has correct row count and timing
- Continuous data arrays match between formats (after calibration)

---

## Usage After Implementation

```bash
# DH5 (default, unchanged)
uv run ephyscon convert session_path

# NWB via config file
cat > nwb_config.json << EOF
{
  "output_format": "nwb",
  "nwb_output_options": {
    "experimenter": ["Your Name"],
    "institution": "Your Institution",
    "lab": "Your Lab",
    "session_description": "Open Ephys recording"
  },
  "decimation_config": {},
  "event_config": {"network_events_offset": 1000},
  "trialmap_config": {}
}
EOF
uv run ephyscon convert session_path --config nwb_config.json --output-folder output/

# Validate NWB file
python -c "
from pynwb import NWBHDF5IO
with NWBHDF5IO('output/session_exp1_rec1.nwb', 'r') as io:
    nwb = io.read()
    print(nwb.session_description)
    print(list(nwb.acquisition.keys()))
"
```

---

## Notes

- `NWBWriter.add_operation()` stores history as JSON in `processing_history` module description field — not ideal but avoids custom NWB extensions
- NWB requires `session_start_time` to be timezone-aware; use `datetime.now().astimezone()`
- BrainBox outcome names (`DH5OutputOptions.add_brainbox_outcome_names`) are DH5-only and ignored by `NWBWriter`
- The `NWBOutputOptions.electrode_metadata` field is defined but not yet wired up — implement when electrode table is added
