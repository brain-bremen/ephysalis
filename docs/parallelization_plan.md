# Parallelization Plan

## Goal

Speed up per-channel processing steps (decimation, continuous MUA) using
`joblib.Parallel`. The global `n_jobs` field on `OpenEphysToDhConfig` controls
the worker count (`1` = single-threaded, existing behaviour; `-1` = all cores).

## Constraint: HDF5 is single-writer

`h5py` / `dh5io` does not support concurrent writes from multiple processes.
Workers must **not** open the final DH5 file. Instead:

1. Each worker computes its channel and writes to a **temporary HDF5 file**.
2. After all workers finish, the main process merges temp files into the final
   DH5 in deterministic order.

## Architecture

```
process_channel(channel, cont_id, tmp_dir, config)
    └── compute(channel, config)          # pure numpy/scipy, CPU-bound
    └── write_cont_block(tmp_h5, cont_id) # writes to tmp_dir/cont_<id>.h5

Parallel(n_jobs)(delayed(process_channel)(...) for each channel)

merge(tmp_dir, final_dh5)
    └── for cont_id in sorted order:
            copy_cont_block(tmp_dir/cont_<id>.h5  →  final_dh5)
    └── cleanup tmp_dir
```

## Steps to implement

### 1. Add `joblib` dependency

```toml
# pyproject.toml
dependencies = [
    ...
    "joblib>=1.4",
]
```

### 2. Refactor `decimation.py`

Extract the per-channel compute into a pure function with no HDF5 side-effects:

```python
def _compute_channel(
    channel_name: str,
    oe_cont,
    config: DecimationConfig,
    global_channel_index: int,
    channel_index: int,
) -> tuple[int, np.ndarray, np.ndarray, float, dict]:
    """Returns (cont_id, decimated_samples, region_index, scaling_factor, channel_info)."""
    ...
```

Replace the inner loop in `decimate_raw_data` with:

```python
results = Parallel(n_jobs=n_jobs)(
    delayed(_compute_channel)(name, oe_cont, config, gi, ci)
    for ci, name, gi in channel_iter
)

tmp_dir = Path(tempfile.mkdtemp())
for cont_id, samples, index, scale, ch_info in results:
    _write_cont_to_tmp(tmp_dir, cont_id, samples, index, scale, ch_info)

_merge_tmp_into_dh5(tmp_dir, dh5file)
```

### 3. Refactor `mua.py` — same pattern

Same split: `_compute_mua_channel` (pure) + parallel dispatch + temp-file merge.

### 4. Helper: temp-file merge

Add `src/ephyscon/parallel_utils.py`:

```python
def write_cont_to_tmp(tmp_dir, cont_id, data, index, sample_period_ns,
                      name, channels, calibration): ...

def merge_tmp_into_dh5(tmp_dir, dh5file, cont_ids): ...
```

`merge_tmp_into_dh5` copies CONT groups from each temp file using
`h5py.File.copy()` and deletes the temp directory on success.

### 5. Pass `n_jobs` through the call chain

`convert_open_ephys_recording_to_dh5()` already receives the top-level config;
it passes `config.n_jobs` to `decimate_raw_data` and `extract_continuous_mua`.

### 6. `raw.py` — skip parallelization

Raw data copying is I/O-bound and already fast; no parallelization needed.

### 7. `trialmap.py` / `events.py` — skip

These are not channel-parallel operations.

## Memory profile

Peak memory per worker ≈ one channel × recording length × dtype size.
For 30 kHz, 1 h, float64: ~900 MB per channel.
With `n_jobs=-1` on a 64-channel probe this could be large — document this in
the GUI tooltip and consider chunked parallelism later (N channels at a time).

## Testing

- Unit test `_compute_channel` directly (no HDF5 needed).
- Integration test: run with `n_jobs=1` and `n_jobs=2` on the `tests/data/Test`
  session and assert byte-identical DH5 output (or numerically equivalent CONT
  blocks).
- Benchmark script: `scripts/benchmark_parallelization.py` — record wall time
  vs. `n_jobs` for a real session.
