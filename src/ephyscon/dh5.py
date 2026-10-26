"""One DH5 (DAQ-HDF5) file per session, built from the outputs in the session's work dir.

Contents (all included runs and all probes), on the same time axis as the NWB file
(:mod:`ephyscon.nwb`; times in ns from the session start, ``SESSION_START_MARGIN_S`` before the
first run):

=====================  =========================================================================
DH5                    content
=====================  =========================================================================
``CONT1``...           with ``raw``: raw AP (wideband) data, one block per channel (ids ``1 +``
                       channel index over all probes)
``CONT1601``...        with ``raw``: analog inputs of the aux stream
``CONT2001``...        LFP, one block per channel (ids ``2001 +`` channel index over all probes)
``CONT3601``           running speed (calibration: cm/s per bit)
``CONT3602``, ``3603``   eye position x, y (uncalibrated volts of the eye tracker outputs)
``CONT4001``...        MUA envelope (``ephyscon mua``), ids ``4001 +`` channel index
``CONT8001``...        CSD (``ephyscon csd``), ids ``8001 +`` channel index; int16 scaled as
                       ``dh_csd`` (calibration = max |CSD| / (32767 - 0.01) A/m^3 per count)
                       All CONT blocks have one INDEX region per run, or more if the stream's
                       clock is off the block's sample period (see :data:`MAX_CLOCK_ERROR_SAMPLES`)
``SPIKE<n>``           one block per sorted unit, ``n = 10000 * probe index + unit id``;
                       timestamps only (``spikeSamples = 0``) with the mean waveform on the peak
                       channel as ``MEAN_WAVEFORM``
``TRIALMAP``           trials of all runs (``StimNo`` = condition)
``EV02``               events (``events/<run>.csv``): TTL edges ``+line`` / ``-line``, network
                       events ``word + 1000``; the code names are attributes (:mod:`ephyscon.events`)
``Operations``         ``nnn_ephyscon_export`` with the processing record (JSON) in its
                       ``Processing`` attribute (:mod:`ephyscon.provenance`)
=====================  =========================================================================

The block ids follow the ranges of ``dhspec.cont.CONT_BLOCK_RANGES``, shared by the lab's
converters (at most 1600 probe channels and 400 analog inputs).

DH5 has no place for stimulus parameters and unit metrics; they are written next to the file
as ``<name>_trials.csv`` (with the ``TrialNo`` of the TRIALMAP) and ``<name>_units.csv`` (with
the SPIKE block id). Calibration values convert to volts, as the DH5 specification says.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ephyscon.lfp import LFP_FILTERING, lfp_folder
from ephyscon.mua import MUA_FILTERING, mua_folder
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
from ephyscon.sync import AUX

if TYPE_CHECKING:
    import h5py
    from dhspec.cont import ContSignalType
    from spikeinterface.core import BaseRecording

    from ephyscon.session_config import SessionConfig
    from ephyscon.sync import RunSync

logger = logging.getLogger(__name__)

try:
    from dhspec.cont import CONT_BLOCK_RANGES, ContBlockGroup
except ImportError:  # dh5io <= 0.4.2; remove with the next dh5io release
    from enum import Enum

    class ContBlockGroup(Enum):  # type: ignore[no-redef]
        RAW = "RAW"
        ANALOG = "ANALOG"
        LFP = "LFP"
        ANALOG_DOWNSAMPLED = "ANALOG_DOWNSAMPLED"
        MUA = "MUA/ESA"
        CSD = "CSD"

    CONT_BLOCK_RANGES = {ContBlockGroup.RAW: (1, 1600), ContBlockGroup.ANALOG: (1601, 2000),  # type: ignore[dict-item]
                         ContBlockGroup.LFP: (2001, 3600), ContBlockGroup.ANALOG_DOWNSAMPLED: (3601, 4000),
                         ContBlockGroup.MUA: (4001, 5600), ContBlockGroup.CSD: (8001, 9600)}

# CONT block ids of the lab's converters (dhspec.cont.CONT_BLOCK_RANGES)
RAW_FIRST_CONT_ID = CONT_BLOCK_RANGES[ContBlockGroup.RAW][0]
ANALOG_FIRST_CONT_ID = CONT_BLOCK_RANGES[ContBlockGroup.ANALOG][0]
LFP_FIRST_CONT_ID = CONT_BLOCK_RANGES[ContBlockGroup.LFP][0]
MUA_FIRST_CONT_ID = CONT_BLOCK_RANGES[ContBlockGroup.MUA][0]
CSD_FIRST_CONT_ID = CONT_BLOCK_RANGES[ContBlockGroup.CSD][0]
RUNNING_SPEED_CONT_ID = CONT_BLOCK_RANGES[ContBlockGroup.ANALOG_DOWNSAMPLED][0]
EYE_CONT_IDS = (RUNNING_SPEED_CONT_ID + 1, RUNNING_SPEED_CONT_ID + 2)
MAX_NEURAL_CHANNELS = CONT_BLOCK_RANGES[ContBlockGroup.RAW][1] - RAW_FIRST_CONT_ID + 1
MAX_ANALOG_CHANNELS = CONT_BLOCK_RANGES[ContBlockGroup.ANALOG][1] - ANALOG_FIRST_CONT_ID + 1
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


def stream_timing(run: Run, key: str, rate: float, offset: float, syncs: dict[str, RunSync]) -> tuple[float, float]:
    """Start (s, session axis) and true rate of a signal at nominal ``rate`` that is
    sample-locked to stream ``key`` (a probe's AP stream, or ``AUX``) of a run."""
    clock = syncs[run.name].maps.get(key) if run.name in syncs else None
    if clock is None:
        return offset, rate
    stream_rate = (run.format.aux_sampling_frequency(run) if key == AUX
                   else run.get_recording(StreamKind.AP, key).sampling_frequency)
    return offset + float(clock.to_reference_time(0.0)), rate * clock.effective_sampling_frequency / stream_rate


def _write_cont_blocks(file: h5py.File, parts: list[tuple[BaseRecording, float, float]], first_id: int,
                       first_global: int, label: str, signal_type: ContSignalType | None, comment: str,
                       to_int16: bool = False, full_scale: float = 32000.0, unit_factor: float = 1e-6,
                       c_rounding: bool = False) -> int:
    """One CONT block per channel of the recordings ``parts`` = ``[(recording, start s, true
    rate)]`` (the runs, in order), ids ``first_id + channel``. int16 recordings are copied with
    their gains as calibration; with ``to_int16`` (float, e.g. MUA in µV) each channel is scaled
    so that its peak is ``full_scale`` counts (calibration in the unit of the traces times
    ``unit_factor``; rounding half away from zero with ``c_rounding``, as ``dh_csd``). Returns
    the number of blocks."""
    import dh5io.cont
    from dhspec.cont import create_empty_index_array

    first = parts[0][0]
    period_ns = int(round(1e9 / first.sampling_frequency))
    stored_rate = 1e9 / period_ns  # the rate a DH5 reader assumes; regions correct the rest
    regions, total = [], 0
    for recording, start, rate in parts:
        regions += [(t, total + k) for t, k in lfp_regions(start, rate, stored_rate, recording.get_num_samples())]
        total += recording.get_num_samples()
    index = create_empty_index_array(len(regions))
    index["time"], index["offset"] = [r[0] for r in regions], [r[1] for r in regions]
    channel_ids = [str(c) for c in first.channel_ids]
    if to_int16:
        peak = np.zeros(len(channel_ids))
        for recording, _, _ in parts:
            n = recording.get_num_samples()
            for begin in range(0, n, CHUNK_SAMPLES):
                traces = recording.get_traces(start_frame=begin, end_frame=min(n, begin + CHUNK_SAMPLES))
                peak = np.maximum(peak, np.nanmax(np.abs(traces), axis=0))
        calibration = np.where(peak > 0, peak, 1.0) * unit_factor / full_scale  # per bit
    else:
        calibration = first.get_channel_gains() * 1e-6
    groups = []
    for channel, channel_id in enumerate(channel_ids):
        group = dh5io.cont.create_empty_cont_group_in_file(
            file, first_id + channel, nSamples=total, nChannels=1, sample_period_ns=np.int32(period_ns),
            n_index_items=len(regions), calibration=np.array([calibration[channel]]),
            channels=_channel_info(first_global + channel, channel), name=f"{label} {channel_id}",
            signal_type=signal_type, comment=comment,
        )
        group["INDEX"][:] = index
        groups.append(group["DATA"])
    written = 0
    for recording, _, _ in parts:
        n = recording.get_num_samples()
        for begin in range(0, n, CHUNK_SAMPLES):
            traces = recording.get_traces(start_frame=begin, end_frame=min(n, begin + CHUNK_SAMPLES), return_in_uV=False)
            if to_int16:
                scaled = np.nan_to_num(traces).astype(np.float64) * unit_factor / calibration
                traces = (np.sign(scaled) * np.floor(np.abs(scaled) + 0.5) if c_rounding else np.round(scaled)).astype(np.int16)
            for channel, data in enumerate(groups):
                data[written + begin: written + begin + len(traces), 0] = traces[:, channel]
        written += n
    return len(channel_ids)


def _probe_channel_counts(session: Session, runs: list[Run]) -> dict[str, int]:
    return {p: len(next(r for r in runs if p in r.probes).get_recording(StreamKind.AP, p).channel_ids)
            for p in session.probe_ids}


def _write_derived(file: h5py.File, session: Session, work_dir: Path, runs: list[Run], offsets: dict[str, float],
                   syncs: dict[str, RunSync], kind: str) -> int:
    """LFP (``kind="lfp"``), MUA (``"mua"``) or CSD (``"csd"``) of all probes from the work dir."""
    import spikeinterface.core as si
    from dhspec.cont import ContSignalType

    from ephyscon.csd import CSD_UNIT, DH5_FULL_SCALE, csd_folder

    folder, first_id, signal_type, comment = {
        "lfp": (lfp_folder, LFP_FIRST_CONT_ID, ContSignalType.LFP, f"LFP: {LFP_FILTERING}"),
        "mua": (mua_folder, MUA_FIRST_CONT_ID, ContSignalType.MUA, f"MUA envelope: {MUA_FILTERING}"),
        "csd": (csd_folder, CSD_FIRST_CONT_ID, ContSignalType.CSD,
                f"CSD of the LFP (dh_csd V3.1 [RBF] port, ephyscon.csd); Calibration in {CSD_UNIT} per count"),
    }[kind]
    scaling: dict[str, Any] = ({"to_int16": True} if kind == "mua" else
               {"to_int16": True, "full_scale": DH5_FULL_SCALE, "unit_factor": 1.0, "c_rounding": True} if kind == "csd"
               else {})
    counts = _probe_channel_counts(session, runs)
    if sum(counts.values()) > MAX_NEURAL_CHANNELS:
        raise ValueError(f"{sum(counts.values())} probe channels do not fit the {kind.upper()} CONT ids")
    global_index, n_blocks = 0, 0
    for probe_id, n_channels in counts.items():
        parts = []
        for run in runs:
            path = folder(work_dir, run.name, probe_id)
            if (path / "binary.json").exists():
                recording = si.load(path)
                parts.append((recording, *stream_timing(run, probe_id, recording.sampling_frequency,
                                                        offsets[run.name], syncs)))
        if parts:
            n_blocks += _write_cont_blocks(
                file, parts, first_id + global_index, global_index, f"{probe_id} {kind.upper()}", signal_type,
                f"{comment}; INDEX regions placed on the session time axis", **scaling)
        global_index += n_channels
    return n_blocks


def _write_raw(file: h5py.File, session: Session, runs: list[Run], offsets: dict[str, float],
               syncs: dict[str, RunSync]) -> int:
    """Raw AP (wideband) data of all probes (CONT 1...) and the analog aux inputs (CONT 1601...)."""
    from dhspec.cont import ContSignalType

    counts = _probe_channel_counts(session, runs)
    if sum(counts.values()) > MAX_NEURAL_CHANNELS:
        raise ValueError(f"{sum(counts.values())} probe channels do not fit the raw CONT ids")
    global_index, n_blocks = 0, 0
    for probe_id, n_channels in counts.items():
        parts = []
        for run in runs:
            if probe_id in run.probes:
                recording = run.get_recording(StreamKind.AP, probe_id)
                parts.append((recording, *stream_timing(run, probe_id, recording.sampling_frequency,
                                                        offsets[run.name], syncs)))
        n_blocks += _write_cont_blocks(file, parts, RAW_FIRST_CONT_ID + global_index, global_index,
                                       f"{probe_id} raw", None, "raw data of the AP (wideband) stream")
        global_index += n_channels
    parts = []
    for run in runs:
        if run.aux_stream is not None:
            recording = run.format.aux_analog_recording(run)
            parts.append((recording, *stream_timing(run, AUX, recording.sampling_frequency, offsets[run.name], syncs)))
    if parts and parts[0][0].get_num_channels() > MAX_ANALOG_CHANNELS:
        raise ValueError(f"{parts[0][0].get_num_channels()} analog inputs do not fit the analog CONT ids")
    if parts:
        n_blocks += _write_cont_blocks(file, parts, ANALOG_FIRST_CONT_ID, -1, "aux", ContSignalType.ANALOG,
                                       "analog input of the aux stream")
    return n_blocks


def _write_events(file: h5py.File, work_dir: Path, runs: list[Run], offsets: dict[str, float],
                  config: SessionConfig | None) -> int:
    """EV02 from ``events/<run>.csv``, with the names of the codes as attributes."""
    import dh5io.event_triggers
    from dhspec.event_triggers import EV_DATASET_NAME

    from ephyscon.events import event_code_names, load_run_events
    from ephyscon.session_config import EventsConfig

    events_config = config.events if config is not None else EventsConfig()
    times, codes, names = [], [], {}
    network = False
    for run in runs:
        path = work_dir / "events" / f"{run.name}.csv"
        if not path.exists():
            continue
        events = load_run_events(path)
        times.append(_ns(events.time.to_numpy(float) + offsets[run.name]))
        codes.append(events.code.to_numpy(np.int32))
        ttl = events[(events.source == "ttl") & (events.name != "")]
        for line, name in ttl.groupby("line").name.first().items():
            names[f"{name}On"], names[f"{name}Off"] = int(line), -int(line)
        network |= bool((events.source == "network").any())
    if not times:
        return 0
    if network:
        names.update(event_code_names(events_config.network_events_offset))
    all_times, all_codes = np.concatenate(times), np.concatenate(codes)
    order = np.argsort(all_times, kind="stable")
    dh5io.event_triggers.add_event_triggers_to_file(file, timestamps_ns=all_times[order].astype(np.int64),
                                                    event_codes=all_codes[order])
    for name, code in names.items():
        file[EV_DATASET_NAME].attrs[name] = np.int32(code)
    return len(all_times)


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


def trial_numbers(trials: pd.DataFrame) -> np.ndarray:
    """TRIALMAP ``TrialNo``: VStim's trial numbers if every trial has one (as the lab's
    converters), else the position in the session."""
    if "vstim_trial" in trials and trials["vstim_trial"].notna().all():
        return trials["vstim_trial"].to_numpy().astype(np.int64)
    return np.arange(len(trials))


def _write_trialmap(file: h5py.File, trials: pd.DataFrame) -> None:
    import dh5io.trialmap
    from dhspec.trialmap import TRIALMAP_DATASET_DTYPE

    trialmap = np.recarray(len(trials), dtype=TRIALMAP_DATASET_DTYPE)
    trialmap.TrialNo = trial_numbers(trials)
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
    processing: dict | None = None,
    raw: bool = False,
) -> Path:
    """Write the DH5 file of a session from the outputs in ``work_dir`` (see the module docs),
    plus ``<name>_trials.csv`` and ``<name>_units.csv`` next to it.

    ``syncs`` maps run names to their :class:`RunSync`; it places the LFP of probes other than
    the reference probe. Missing outputs (e.g. no sorting yet) are skipped. ``processing`` is
    the processing record (:func:`ephyscon.provenance.processing_record`). ``raw`` adds the raw
    data (large: about the size of the recording).
    """
    import dh5io.operations
    from dh5io.create import create_dh_file
    from dh5io.validation import validate_dh5_file

    from ephyscon.provenance import DH5_ATTRIBUTE_NAME, DH5_OPERATION_NAME
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
        n_raw = _write_raw(file, session, runs, offsets, syncs or {}) if raw else 0
        n_lfp = _write_derived(file, session, work_dir, runs, offsets, syncs or {}, "lfp") if lfp else 0
        n_mua = _write_derived(file, session, work_dir, runs, offsets, syncs or {}, "mua")
        n_csd = _write_derived(file, session, work_dir, runs, offsets, syncs or {}, "csd")
        n_events = _write_events(file, work_dir, runs, offsets, config)
        _write_running_speed(file, work_dir, runs, offsets)
        _write_eye_position(file, work_dir, runs, offsets)
        units = _write_spikes(file, session, work_dir, runs, offsets)
        if not trials.empty:
            _write_trialmap(file, trials)
        dh5io.operations.add_operation_to_file(file, DH5_OPERATION_NAME, tool=f"ephyscon {get_version_from_pyproject()}",
                                               date=datetime.now())
        if processing is not None:
            operation = file["Operations"][sorted(file["Operations"])[-1]]
            operation.attrs[DH5_ATTRIBUTE_NAME] = json.dumps(processing, default=str)
        validate_dh5_file(file)
    tmp.replace(path)
    sidecars = sidecar_paths(path)
    if not trials.empty:
        trials.insert(0, "TrialNo", trial_numbers(trials))
        trials.to_csv(sidecars["trials"], index=False)
    if not units.empty:
        units.to_csv(sidecars["units"], index=False)
    logger.info("wrote %s (%d raw, %d LFP, %d MUA, %d CSD blocks, %d units, %d trials, %d events)", path, n_raw,
                n_lfp, n_mua, n_csd, len(units), len(trials), n_events)
    return path
