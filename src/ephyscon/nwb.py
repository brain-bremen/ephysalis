"""One NWB file per session, built from the outputs in the session's work dir.

Contents (all included runs and all probes):

==========================  ===============================================================
NWB                         source
==========================  ===============================================================
devices, electrode groups   probe metadata of the recording files, ``session.toml``
electrodes                  channel positions of the AP streams; ``in_brain`` (closer to the
                            tip than ``insertion_depth_um``), ``bad_channel``
epochs                      one per run (tag = run name)
trials                      ``trials/<run>_<paradigm>.csv`` of all runs, with ``run`` and
                            ``paradigm`` columns; parameters missing in a paradigm are NaN
units                       ``sorting/<probe>/units.csv`` and ``spikes/<run>_<probe>.npz``;
                            mean waveform on the peak channel from the SortingAnalyzer
processing/ecephys/LFP      ``lfp/<run>_<probe>/``, one ElectricalSeries per run and probe
processing/behavior         running speed per run (``running_speed/<run>.npy``)
==========================  ===============================================================

Time base: seconds from ``session_start_time``, which is 1 s before the start of the first
included run, so that streams starting slightly before the reference probe still have
positive times. The runs are separate recordings, so a run's offset is the difference of
their start times (SpikeGLX ``fileCreateTime``, 1 s resolution). Within a run, times are on the reference probe's clock,
the clock of the trial times and spike tables; the LFP of other probes is mapped onto it with
the linear fit of the run's sync clock map.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from hdmf.data_utils import GenericDataChunkIterator

from ephyscon.lfp import lfp_folder
from ephyscon.session import Run, Session, StreamKind
from ephyscon.sorting import load_run_spikes, sorting_channels

if TYPE_CHECKING:
    from pynwb import NWBFile
    from spikeinterface.core import BaseRecording

    from ephyscon.session_config import SessionConfig
    from ephyscon.sync import RunSync

logger = logging.getLogger(__name__)

SESSION_START_MARGIN_S = 1.0  # session_start_time lies this long before the first run
RUNNING_SPEED_BIN_S = 0.001  # running speed sample k is the mean over (k, k+1] ms

UNIT_COLUMN_DESCRIPTIONS = {
    "probe": "probe id",
    "unit_id": "unit id in the probe's Kilosort output",
    "KSLabel": "Kilosort label: good (refractory period respected) or mua",
    "ContamPct": "Kilosort contamination estimate (%)",
    "Amplitude": "Kilosort template amplitude",
    "x_um": "unit position across the probe (um, monopolar triangulation)",
    "y_um": "unit position along the probe from the tip (um, monopolar triangulation)",
}


class RecordingDataChunkIterator(GenericDataChunkIterator):
    """Writes a spikeinterface recording (samples x channels, unscaled) chunk by chunk."""

    def __init__(self, recording: BaseRecording, **kwargs: Any):
        self.recording = recording
        super().__init__(**kwargs)

    def _get_data(self, selection: tuple[slice, ...]) -> np.ndarray:
        frames, channels = selection
        traces = self.recording.get_traces(start_frame=frames.start, end_frame=frames.stop, return_in_uV=False)
        return traces[:, channels]

    def _get_dtype(self) -> np.dtype:
        return np.dtype(self.recording.get_dtype())

    def _get_maxshape(self) -> tuple[int, int]:
        return (self.recording.get_num_samples(), self.recording.get_num_channels())


def included_runs(session: Session, config: SessionConfig | None) -> list[Run]:
    return [
        r for r in session.runs
        if config is None or (rc := config.run_config(r.name)) is None or rc.include
    ]


def run_offsets(runs: list[Run]) -> dict[str, float]:
    """Start of each run in seconds from the session start (``SESSION_START_MARGIN_S`` before
    the first run)."""
    if any(r.start_time is None for r in runs):
        raise ValueError("all runs need a start time to place them on one time axis")
    t0 = runs[0].start_time
    offsets = {r.name: SESSION_START_MARGIN_S + (r.start_time - t0).total_seconds() for r in runs}  # type: ignore[operator]
    for a, b in zip(runs, runs[1:]):
        if offsets[a.name] + (a.duration_s or 0.0) > offsets[b.name]:
            logger.warning("runs %s and %s overlap on the session time axis", a.name, b.name)
    return offsets


def _paradigm(csv: Path, run_name: str) -> str:
    return csv.stem.removeprefix(run_name + "_")


def session_trials(work_dir: Path, runs: list[Run], offsets: dict[str, float]) -> pd.DataFrame:
    """Trials of all runs on the session time axis, with ``run`` and ``paradigm`` columns."""
    tables = []
    for run in runs:
        for csv in sorted((work_dir / "trials").glob(f"{run.name}_*.csv")):
            trials = pd.read_csv(csv)
            trials.insert(0, "paradigm", _paradigm(csv, run.name))
            trials.insert(0, "run", run.name)
            trials[["start_time", "stop_time"]] += offsets[run.name]
            tables.append(trials)
    if not tables:
        return pd.DataFrame()
    table = pd.concat(tables, ignore_index=True).sort_values("start_time", kind="stable")
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].fillna("").astype(str)
    return table.reset_index(drop=True)


def _new_nwbfile(session: Session, config: SessionConfig | None, start_time: datetime) -> NWBFile:
    from pynwb import NWBFile
    from pynwb.file import Subject

    from ephyscon.version import get_version_from_pyproject

    subject = None
    if config is not None:
        s = config.subject
        subject = Subject(
            subject_id=config.subject_id, species=s.species, sex=s.sex, age=s.age, strain=s.strain,
            genotype=s.genotype, description=s.description or None,
            date_of_birth=datetime.fromisoformat(s.date_of_birth).astimezone() if s.date_of_birth else None,
        )
    return NWBFile(
        session_description=(config.description if config and config.description else session.name),
        identifier=str(uuid.uuid4()),
        session_start_time=start_time.astimezone(),
        session_id=config.session_id if config else session.name,
        experimenter=(config.experimenter or None) if config else None,
        lab=config.lab if config else None,
        institution=config.institution if config else None,
        notes=__doc__.split("Time base: ", 1)[1].strip(),
        source_script=f"ephyscon {get_version_from_pyproject()}",
        source_script_file_name="ephyscon",
        subject=subject,
    )


def _add_electrodes(nwbfile: NWBFile, session: Session, runs: list[Run], config: SessionConfig | None) -> dict:
    """Devices, electrode groups and electrodes; returns ``{(probe, channel index): row}``."""
    nwbfile.add_electrode_column("channel_name", "channel id in the AP stream")
    nwbfile.add_electrode_column("in_brain", "contact closer to the tip than the insertion depth")
    nwbfile.add_electrode_column("bad_channel", "excluded from sorting (session.toml bad_channels)")
    rows: dict[tuple[str, int], int] = {}
    for probe_id in session.probe_ids:
        run = next(r for r in runs if probe_id in r.probes)
        info = run.probes[probe_id]
        probe_config = config.probes.get(probe_id) if config else None
        model_name = info.part_number or info.model_name or "Neuropixels"
        model = nwbfile.device_models.get(model_name) or nwbfile.create_device_model(
            name=model_name, manufacturer="IMEC", model_number=info.part_number,
            description=info.model_name or f"Neuropixels probe, part number {info.part_number or 'unknown'}",
        )
        device = nwbfile.create_device(
            name=probe_id, description=f"{info.model_name or 'Neuropixels'} ({model_name})",
            serial_number=info.serial_number, model=model,
        )
        location = probe_config.area if probe_config and probe_config.area else "unknown"
        group = nwbfile.create_electrode_group(
            name=probe_id, description=f"{probe_id}, {info.num_shanks} shank(s)", location=location, device=device,
        )
        recording = run.get_recording(StreamKind.AP, probe_id)
        in_brain = set(sorting_channels(recording, probe_config and probe_config.model_copy(update={"bad_channels": []})))
        bad = set(probe_config.bad_channels) if probe_config else set()
        for index, (channel_id, (x, y)) in enumerate(zip(recording.channel_ids, recording.get_channel_locations())):
            rows[(probe_id, index)] = len(nwbfile.electrodes) if nwbfile.electrodes is not None else 0
            nwbfile.add_electrode(
                group=group, location=location, rel_x=float(x), rel_y=float(y), channel_name=str(channel_id),
                in_brain=channel_id in in_brain, bad_channel=index in bad,
            )
    return rows


def _add_epochs_and_trials(nwbfile: NWBFile, work_dir: Path, runs: list[Run], offsets: dict[str, float]) -> None:
    for run in runs:
        nwbfile.add_epoch(start_time=offsets[run.name], stop_time=offsets[run.name] + (run.duration_s or 0.0),
                          tags=[run.name])
    trials = session_trials(work_dir, runs, offsets)
    if trials.empty:
        return
    columns = [c for c in trials.columns if c not in ("start_time", "stop_time")]
    for column in columns:
        nwbfile.add_trial_column(column, {"run": "run name", "paradigm": "paradigm of the stimulus system",
                                          "trial": "trial index within the run and paradigm"}.get(
                                              column, "stimulus parameter (NaN if the paradigm does not have it)"))
    for values in trials.to_dict("records"):  # not itertuples: it renames columns that are not identifiers
        nwbfile.add_trial(**values)


def _add_units(nwbfile: NWBFile, work_dir: Path, runs: list[Run], offsets: dict[str, float],
               electrode_rows: dict, channel_index: dict[str, dict[str, int]]) -> None:
    from pynwb.misc import Units

    probes = [p for p in channel_index if (work_dir / "sorting" / p / "units.csv").exists()]
    if not probes:
        return
    tables, waveforms, fs = {}, {}, None
    for probe_id in probes:
        tables[probe_id] = pd.read_csv(work_dir / "sorting" / probe_id / "units.csv")
        waveforms[probe_id], rate = _peak_waveforms(work_dir / "sorting" / probe_id / "analyzer", tables[probe_id])
        fs = fs or rate
    nwbfile.units = Units(
        name="units", description="Kilosort4 units (uncurated), all probes; spike times on the session time axis",
        electrode_table=nwbfile.electrodes, waveform_rate=fs, waveform_unit="volts",
        resolution=1 / fs if fs else -1.0,
    )
    extra = [c for c in tables[probes[0]].columns if c != "peak_channel"]
    nwbfile.add_unit_column("probe", UNIT_COLUMN_DESCRIPTIONS["probe"])
    for column in extra:
        nwbfile.add_unit_column(column, UNIT_COLUMN_DESCRIPTIONS.get(
            column, "SortingAnalyzer quality or template metric (spikeinterface)"))

    for probe_id in probes:
        units = tables[probe_id]
        times, ids, intervals = [], [], []
        for run in runs:
            path = work_dir / "spikes" / f"{run.name}_{probe_id}.npz"
            if path.exists():
                spikes = load_run_spikes(path)
                times.append(spikes.time.to_numpy() + offsets[run.name])
                ids.append(spikes.unit_id.to_numpy())
                intervals.append([offsets[run.name], offsets[run.name] + (run.duration_s or 0.0)])
        all_times, all_ids = np.concatenate(times), np.concatenate(ids)
        order = np.lexsort((all_times, all_ids))
        all_times, all_ids = all_times[order], all_ids[order]
        bounds = np.searchsorted(all_ids, units.unit_id.to_numpy(), side="left"), \
            np.searchsorted(all_ids, units.unit_id.to_numpy(), side="right")
        probe_waveforms = waveforms[probe_id]
        for k, values in enumerate(units.to_dict("records")):
            row = electrode_rows[(probe_id, channel_index[probe_id][values.pop("peak_channel")])]
            if probe_waveforms is not None:
                values["waveform_mean"] = probe_waveforms[k]
            nwbfile.add_unit(
                spike_times=all_times[bounds[0][k]:bounds[1][k]], electrodes=[row],
                electrode_group=nwbfile.electrode_groups[probe_id], obs_intervals=intervals, probe=probe_id,
                **values,
            )


def _peak_waveforms(analyzer_dir: Path, units: pd.DataFrame) -> tuple[np.ndarray | None, float | None]:
    """Mean waveform of each unit on its peak channel (volts), in the order of ``units``."""
    if not analyzer_dir.exists():
        return None, None
    import spikeinterface.core as si

    analyzer = si.load_sorting_analyzer(analyzer_dir, load_extensions=False)
    templates = analyzer.load_extension("templates").get_data(operator="average")  # units x samples x channels, uV
    unit_pos = {u: i for i, u in enumerate(analyzer.unit_ids)}
    channel_pos = {str(c): i for i, c in enumerate(analyzer.channel_ids)}
    waveforms = np.stack([
        templates[unit_pos[u], :, channel_pos[c]] for u, c in zip(units.unit_id, units.peak_channel)
    ]) * 1e-6
    return waveforms, float(analyzer.sampling_frequency)


def _add_lfp(nwbfile: NWBFile, work_dir: Path, runs: list[Run], offsets: dict[str, float],
             syncs: dict[str, RunSync], electrode_rows: dict, compression: str | None) -> None:
    import spikeinterface.core as si
    from hdmf.backends.hdf5 import H5DataIO
    from pynwb.ecephys import LFP, ElectricalSeries

    module = nwbfile.create_processing_module("ecephys", "LFP per run and probe (1 kHz, anti-alias filtered)")
    container = LFP(name="LFP")
    module.add(container)
    for run in runs:
        for probe_id in run.probes:
            folder = lfp_folder(work_dir, run.name, probe_id)
            if not (folder / "binary.json").exists():
                continue
            recording = si.load(folder)
            ap = run.get_recording(StreamKind.AP, probe_id)
            # LFP sample k is AP sample k * fs_ap / fs_lfp; map that onto the reference clock (linear fit)
            clock = syncs[run.name].maps[probe_id] if run.name in syncs else None
            samples_per_lfp = ap.sampling_frequency / recording.sampling_frequency
            if clock is not None:
                start = float(clock.to_reference_time(0.0))  # before the first sync edge: the linear fit
                rate = recording.sampling_frequency * clock.effective_sampling_frequency / ap.sampling_frequency
            else:
                start, rate = 0.0, recording.sampling_frequency
            gains, offsets_uv = recording.get_channel_gains(), recording.get_channel_offsets()  # uV per bit, uV
            uniform = bool(np.allclose(gains, gains[0]))
            region = nwbfile.create_electrode_table_region(
                [electrode_rows[(probe_id, i)] for i in range(recording.get_num_channels())],
                f"{probe_id} electrodes",
            )
            data = RecordingDataChunkIterator(recording, buffer_gb=0.2, display_progress=False)
            container.add_electrical_series(ElectricalSeries(
                name=f"LFP_{run.name}_{probe_id}",
                data=H5DataIO(data, compression=compression) if compression else data,
                electrodes=region,
                starting_time=offsets[run.name] + start,
                rate=rate,
                conversion=1e-6 * float(gains[0]) if uniform else 1e-6,
                channel_conversion=None if uniform else gains,
                offset=1e-6 * float(offsets_uv[0]),
                filtering="polyphase resampling to 1 kHz (spikeinterface resample)",
                description=f"LFP of {probe_id} in run {run.name} ({samples_per_lfp:g} AP samples per sample)",
            ))


def _add_running_speed(nwbfile: NWBFile, work_dir: Path, runs: list[Run], offsets: dict[str, float]) -> None:
    from pynwb import TimeSeries
    from pynwb.behavior import BehavioralTimeSeries

    series = []
    for run in runs:
        path = work_dir / "running_speed" / f"{run.name}.npy"
        if path.exists():
            series.append(TimeSeries(
                name=f"running_speed_{run.name}", data=np.load(path), unit="cm/s",
                starting_time=offsets[run.name] + RUNNING_SPEED_BIN_S / 2, rate=1 / RUNNING_SPEED_BIN_S,
                description="running speed from the wheel encoder, mean over 1 ms bins (time = bin centre)",
            ))
    if series:
        module = nwbfile.create_processing_module("behavior", "behavioural signals")
        module.add(BehavioralTimeSeries(time_series=series, name="RunningSpeed"))


def write_session_nwb(
    session: Session,
    work_dir: Path,
    path: Path,
    config: SessionConfig | None = None,
    syncs: dict[str, RunSync] | None = None,
    lfp: bool = True,
    compression: str | None = "gzip",
) -> Path:
    """Write the NWB file of a session from the outputs in ``work_dir`` (see the module docs).

    ``syncs`` maps run names to their :class:`RunSync`; it is needed to place the LFP of
    probes other than the reference probe. Missing outputs (e.g. no sorting yet) are skipped.
    """
    from pynwb import NWBHDF5IO

    runs = included_runs(session, config)
    offsets = run_offsets(runs)
    assert runs[0].start_time is not None  # checked by run_offsets
    nwbfile = _new_nwbfile(session, config, runs[0].start_time - timedelta(seconds=SESSION_START_MARGIN_S))
    electrode_rows = _add_electrodes(nwbfile, session, runs, config)
    channel_index = {
        p: {str(c): i for i, c in enumerate(next(r for r in runs if p in r.probes).get_recording(StreamKind.AP, p).channel_ids)}
        for p in session.probe_ids
    }
    _add_epochs_and_trials(nwbfile, work_dir, runs, offsets)
    _add_units(nwbfile, work_dir, runs, offsets, electrode_rows, channel_index)
    _add_running_speed(nwbfile, work_dir, runs, offsets)
    if lfp:
        _add_lfp(nwbfile, work_dir, runs, offsets, syncs or {}, electrode_rows, compression)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.nwb")
    with NWBHDF5IO(str(tmp), "w") as io:
        io.write(nwbfile)
    tmp.replace(path)
    logger.info("wrote %s", path)
    return path
