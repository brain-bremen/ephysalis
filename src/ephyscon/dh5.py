"""One DH5 (DAQ-HDF5) file per session, built from the outputs in the session's work dir.

Contents (all included runs and all probes), on the same time axis as the NWB file
(:mod:`ephyscon.nwb`; times in ns from the session start, ``SESSION_START_MARGIN_S`` before the
first run):

=====================  =========================================================================
DH5                    content
=====================  =========================================================================
``CONT2001``...        LFP, one block per channel (ids ``2001 +`` channel index over all probes,
                       as ``ephyscon convert``); one INDEX region per run, or more if the probe's
                       clock is off its nominal rate (see :data:`MAX_CLOCK_ERROR_SAMPLES`)
``CONT6001``           running speed (calibration: cm/s per bit)
``CONT6002``, ``6003``   eye position x, y (uncalibrated volts of the eye tracker outputs)
``SPIKE<n>``           one block per sorted unit, ``n = 10000 * probe index + unit id``;
                       timestamps only (``spikeSamples = 0``) with the mean waveform on the peak
                       channel as ``MEAN_WAVEFORM``
``TRIALMAP``           trials of all runs (``StimNo`` = condition)
=====================  =========================================================================

DH5 has no place for stimulus parameters and unit metrics; they are written next to the file
as ``<name>_trials.csv`` (with the ``TrialNo`` of the TRIALMAP) and ``<name>_units.csv`` (with
the SPIKE block id). Calibration values convert to volts, as the DH5 specification says.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ephyscon.lfp import lfp_folder
from ephyscon.nwb import (
    RUNNING_SPEED_BIN_S,
    SESSION_START_MARGIN_S,
    included_runs,
    run_offsets,
    session_trials,
    sorting_info,
)
from ephyscon.session import Run, Session, StreamKind
from ephyscon.sorting import load_run_spikes

if TYPE_CHECKING:
    import h5py

    from ephyscon.session_config import SessionConfig
    from ephyscon.sync import RunSync

logger = logging.getLogger(__name__)

LFP_FIRST_CONT_ID = 2001
RUNNING_SPEED_CONT_ID = 6001
EYE_CONT_IDS = (6002, 6003)
SPIKE_IDS_PER_PROBE = 10000
# A DH5 CONT block has one sample period; a probe whose clock runs off the nominal rate gets a
# new INDEX region (with its exact start time) before the accumulated error reaches this
MAX_CLOCK_ERROR_SAMPLES = 0.1
TRIAL_OUTCOME_HIT = 1  # dhspec / VStim TrialOutcome.Hit, for trials without an outcome
CHUNK_SAMPLES = 60_000


def _ns(seconds: float | np.ndarray) -> np.ndarray:
    return np.round(np.asarray(seconds) * 1e9).astype(np.int64)


def _channel_info(global_index: int, board_index: int) -> np.ndarray:
    from dhspec.cont import create_channel_info

    return np.atleast_1d(create_channel_info(  # one-element array: a scalar would be stored as a scalar attribute
        GlobalChanNumber=global_index, BoardChanNo=board_index, ADCBitWidth=16,
        MaxVoltageRange=0.0, MinVoltageRange=0.0, AmplifChan0=0,
    ))


def lfp_regions(start_s: float, rate: float, nominal_rate: float, num_samples: int) -> list[tuple[int, int]]:
    """INDEX regions ``(time in ns, first sample)`` of one run of a CONT block with
    ``nominal_rate``, for samples that are really ``rate`` apart, starting at ``start_s``."""
    error_per_sample = abs(1 / rate - 1 / nominal_rate) * nominal_rate  # in samples
    length = num_samples if error_per_sample == 0 else max(1, int(MAX_CLOCK_ERROR_SAMPLES / error_per_sample))
    return [(int(_ns(start_s + k / rate)), k) for k in range(0, num_samples, length)]


def _lfp_timing(run: Run, probe_id: str, lfp_rate: float, offset: float, syncs: dict[str, RunSync]) -> tuple[float, float]:
    """Start (s, session axis) and true rate of the LFP of a run on the reference clock."""
    clock = syncs[run.name].maps[probe_id] if run.name in syncs else None
    if clock is None:
        return offset, lfp_rate
    ap_rate = run.get_recording(StreamKind.AP, probe_id).sampling_frequency
    return (offset + float(clock.to_reference_time(0.0)),
            lfp_rate * clock.effective_sampling_frequency / ap_rate)


def _write_lfp(file: h5py.File, session: Session, work_dir: Path, runs: list[Run], offsets: dict[str, float],
               syncs: dict[str, RunSync]) -> int:
    import dh5io.cont
    import spikeinterface.core as si
    from dhspec.cont import ContSignalType, create_empty_index_array

    global_index, n_blocks = 0, 0
    for probe_id in session.probe_ids:
        parts = [(run, si.load(lfp_folder(work_dir, run.name, probe_id))) for run in runs
                 if (lfp_folder(work_dir, run.name, probe_id) / "binary.json").exists()]
        if not parts:
            global_index += len(next(r for r in runs if probe_id in r.probes).get_recording(StreamKind.AP, probe_id).channel_ids)
            continue
        first = parts[0][1]
        nominal = first.sampling_frequency
        gains, channel_ids = first.get_channel_gains(), [str(c) for c in first.channel_ids]
        regions, total = [], 0
        for run, recording in parts:
            start, rate = _lfp_timing(run, probe_id, nominal, offsets[run.name], syncs)
            regions += [(t, total + k) for t, k in lfp_regions(start, rate, nominal, recording.get_num_samples())]
            total += recording.get_num_samples()
        index = create_empty_index_array(len(regions))
        index["time"], index["offset"] = [r[0] for r in regions], [r[1] for r in regions]
        groups = []
        for channel, channel_id in enumerate(channel_ids):
            group = dh5io.cont.create_empty_cont_group_in_file(
                file, LFP_FIRST_CONT_ID + global_index + channel, nSamples=total, nChannels=1,
                sample_period_ns=np.int32(round(1e9 / nominal)), n_index_items=len(regions),
                calibration=np.array([gains[channel] * 1e-6]), channels=_channel_info(global_index + channel, channel),
                name=f"{probe_id} {channel_id} LFP", signal_type=ContSignalType.LFP,
                comment="LFP (spikeinterface resample); INDEX regions placed on the session time axis",
            )
            group["INDEX"][:] = index
            groups.append(group["DATA"])
        written = 0
        for _, recording in parts:
            n = recording.get_num_samples()
            for start in range(0, n, CHUNK_SAMPLES):
                traces = recording.get_traces(start_frame=start, end_frame=min(n, start + CHUNK_SAMPLES), return_in_uV=False)
                for channel, data in enumerate(groups):
                    data[written + start: written + start + len(traces), 0] = traces[:, channel]
            written += n
        global_index += len(channel_ids)
        n_blocks += len(channel_ids)
    return n_blocks


def _write_running_speed(file: h5py.File, work_dir: Path, runs: list[Run], offsets: dict[str, float]) -> bool:
    import dh5io.cont
    from dhspec.cont import ContSignalType, create_empty_index_array

    parts = [(run, np.load(work_dir / "running_speed" / f"{run.name}.npy")) for run in runs
             if (work_dir / "running_speed" / f"{run.name}.npy").exists()]
    if not parts:
        return False
    speed = np.concatenate([p[1] for p in parts])
    scale = max(float(np.nanmax(np.abs(speed))), 1e-6) / 32000  # cm/s per bit
    index = create_empty_index_array(len(parts))
    index["time"] = [int(_ns(offsets[run.name] + RUNNING_SPEED_BIN_S / 2)) for run, _ in parts]
    index["offset"] = np.cumsum([0] + [len(p[1]) for p in parts[:-1]])
    dh5io.cont.create_cont_group_from_data_in_file(
        file, RUNNING_SPEED_CONT_ID, data=np.round(np.nan_to_num(speed) / scale).astype(np.int16)[:, None],
        index=index, sample_period_ns=np.int32(round(RUNNING_SPEED_BIN_S * 1e9)), calibration=np.array([scale]),
        channels=_channel_info(-1, 0), name="running speed",
        comment="running speed from the wheel encoder, mean over 1 ms bins; calibration gives cm/s, not volts",
        signal_type=ContSignalType.ANALOG,
    )
    return True


def _write_eye_position(file: h5py.File, work_dir: Path, runs: list[Run], offsets: dict[str, float]) -> bool:
    import dh5io.cont
    from dhspec.cont import ContSignalType, create_empty_index_array

    parts = [(run, np.load(work_dir / "eye" / f"{run.name}.npy")) for run in runs
             if (work_dir / "eye" / f"{run.name}.npy").exists()]
    if not parts:
        return False
    eye = np.concatenate([p[1] for p in parts])
    index = create_empty_index_array(len(parts))
    index["time"] = [int(_ns(offsets[run.name] + RUNNING_SPEED_BIN_S / 2)) for run, _ in parts]
    index["offset"] = np.cumsum([0] + [len(p[1]) for p in parts[:-1]])
    for axis, (cont_id, name) in enumerate(zip(EYE_CONT_IDS, ("x", "y"))):
        values = np.nan_to_num(eye[:, axis])
        scale = max(float(np.abs(values).max()), 1e-6) / 32000  # V per bit
        dh5io.cont.create_cont_group_from_data_in_file(
            file, cont_id, data=np.round(values / scale).astype(np.int16)[:, None], index=index,
            sample_period_ns=np.int32(round(RUNNING_SPEED_BIN_S * 1e9)), calibration=np.array([scale]),
            channels=_channel_info(-1, axis), name=f"eye position {name}",
            comment="eye tracker analog output (uncalibrated volts), mean over 1 ms bins",
            signal_type=ContSignalType.ANALOG,
        )
    return True


def _write_spikes(file: h5py.File, session: Session, work_dir: Path, runs: list[Run],
                  offsets: dict[str, float]) -> pd.DataFrame:
    import dh5io.spike

    from ephyscon.nwb import peak_waveforms

    tables = []
    for probe_index, probe_id in enumerate(session.probe_ids):
        units_csv = work_dir / "sorting" / probe_id / "units.csv"
        if not units_csv.exists():
            continue
        units = pd.read_csv(units_csv)
        if (units.unit_id >= SPIKE_IDS_PER_PROBE).any():
            raise ValueError(f"{probe_id}: unit ids >= {SPIKE_IDS_PER_PROBE} do not fit the SPIKE block ids")
        waveforms, fs, nbefore = peak_waveforms(work_dir / "sorting" / probe_id / "analyzer", units)
        sorter = sorting_info(work_dir, probe_id).get("sorter", "kilosort4")
        ap = next(r for r in runs if probe_id in r.probes).get_recording(StreamKind.AP, probe_id)
        channel_index = {str(c): i for i, c in enumerate(ap.channel_ids)}
        first_global = sum(len(next(r for r in runs if p in r.probes).get_recording(StreamKind.AP, p).channel_ids)
                           for p in session.probe_ids[:probe_index])

        times, ids = [], []
        for run in runs:
            path = work_dir / "spikes" / f"{run.name}_{probe_id}.npz"
            if path.exists():
                spikes = load_run_spikes(path)
                times.append(spikes.time.to_numpy() + offsets[run.name])
                ids.append(spikes.unit_id.to_numpy())
        all_times, all_ids = np.concatenate(times), np.concatenate(ids)
        order = np.lexsort((all_times, all_ids))
        all_times, all_ids = all_times[order], all_ids[order]

        block_ids = []
        for k, unit in enumerate(units.to_dict("records")):
            unit_times = all_times[np.searchsorted(all_ids, unit["unit_id"]):np.searchsorted(all_ids, unit["unit_id"], "right")]
            block_id = SPIKE_IDS_PER_PROBE * probe_index + int(unit["unit_id"])
            mean = None if waveforms is None else waveforms[k]  # volts
            scale = max(float(np.abs(mean).max()), 1e-9) / 32000 if mean is not None else 1e-6
            board = channel_index[unit["peak_channel"]]
            label = f", {sorter} label {unit['KSLabel']}" if "KSLabel" in unit else ""
            dh5io.spike.create_spike_group_from_data_in_file(
                file, block_id, _ns(unit_times), sample_period_ns=np.int32(round(1e9 / (fs or ap.sampling_frequency))),
                n_channels=1, pre_trigger_samples=nbefore, calibration=np.array([scale]),
                channels=_channel_info(first_global + board, board),
                mean_waveform=None if mean is None else np.round(mean / scale).astype(np.int16)[:, None],
                name=f"{probe_id} unit {unit['unit_id']}",
                comment=f"{sorter} unit {unit['unit_id']} of {probe_id}, peak channel {unit['peak_channel']}{label}; "
                        "timestamps on the session time axis, no per-spike waveforms",
            )
            block_ids.append(block_id)
        table = units.copy()
        table.insert(0, "probe", probe_id)
        table.insert(0, "spike_block", block_ids)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _write_trialmap(file: h5py.File, trials: pd.DataFrame) -> None:
    import dh5io.trialmap
    from dhspec.trialmap import TRIALMAP_DATASET_DTYPE

    trialmap = np.recarray(len(trials), dtype=TRIALMAP_DATASET_DTYPE)
    trialmap.TrialNo = np.arange(len(trials))
    trialmap.StimNo = trials["condition"].to_numpy() if "condition" in trials else 0
    trialmap.Outcome = trials["outcome"].to_numpy() if "outcome" in trials else TRIAL_OUTCOME_HIT
    trialmap.StartTime = _ns(trials.start_time.to_numpy())
    trialmap.EndTime = _ns(trials.stop_time.to_numpy())
    dh5io.trialmap.add_trialmap_to_file(file, trialmap)


def sidecar_paths(path: Path) -> dict[str, Path]:
    return {kind: path.with_name(f"{path.stem}_{kind}.csv") for kind in ("trials", "units")}


def write_session_dh5(
    session: Session,
    work_dir: Path,
    path: Path,
    config: SessionConfig | None = None,
    syncs: dict[str, RunSync] | None = None,
    lfp: bool = True,
) -> Path:
    """Write the DH5 file of a session from the outputs in ``work_dir`` (see the module docs),
    plus ``<name>_trials.csv`` and ``<name>_units.csv`` next to it.

    ``syncs`` maps run names to their :class:`RunSync`; it places the LFP of probes other than
    the reference probe. Missing outputs (e.g. no sorting yet) are skipped.
    """
    import dh5io.operations
    from dh5io.create import create_dh_file
    from dh5io.validation import validate_dh5_file

    from ephyscon.version import get_version_from_pyproject

    runs = included_runs(session, config)
    offsets = run_offsets(runs)
    assert runs[0].start_time is not None  # checked by run_offsets
    session_start = runs[0].start_time - timedelta(seconds=SESSION_START_MARGIN_S)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.dh5")
    trials = session_trials(work_dir, runs, offsets)
    with create_dh_file(tmp, overwrite=True, boards=session.probe_ids) as dh5file:
        file = dh5file._file
        file.attrs["SessionStartTime"] = session_start.isoformat()
        n_lfp = _write_lfp(file, session, work_dir, runs, offsets, syncs or {}) if lfp else 0
        _write_running_speed(file, work_dir, runs, offsets)
        _write_eye_position(file, work_dir, runs, offsets)
        units = _write_spikes(file, session, work_dir, runs, offsets)
        if not trials.empty:
            _write_trialmap(file, trials)
        dh5io.operations.add_operation_to_file(file, "ephyscon_export", tool=f"ephyscon {get_version_from_pyproject()}")
        validate_dh5_file(file)
    tmp.replace(path)
    sidecars = sidecar_paths(path)
    if not trials.empty:
        trials.insert(0, "TrialNo", np.arange(len(trials)))
        trials.to_csv(sidecars["trials"], index=False)
    if not units.empty:
        units.to_csv(sidecars["units"], index=False)
    logger.info("wrote %s (%d LFP blocks, %d units, %d trials)", path, n_lfp, len(units), len(trials))
    return path
