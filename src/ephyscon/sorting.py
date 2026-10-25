"""Spike sorting per probe over the concatenated runs of a session, and cutting the result
back into runs.

Steps (:func:`sort_probe`):

1. Select the channels inside the brain (contacts less than ``insertion_depth_um`` from the
   tip, minus ``bad_channels``) and concatenate the AP streams of all included runs.
2. Write the concatenation to a local binary file once (the raw files are usually on a
   network share, and Kilosort reads the data several times).
3. Run the spike sorter (``[sorting]`` in ``session.toml``, default Kilosort4; any sorter of
   ``spikeinterface.sorters``) on the local copy. Kilosort4 gets checkpoints after its long
   steps (:mod:`ephyscon.kilosort_checkpoints`) so that an interrupted run resumes. Then build
   a SortingAnalyzer (waveforms, templates, unit locations, quality metrics) on the band-pass
   filtered local copy, while it still exists; the sorters filter internally, but wideband
   streams (Neuropixels 2.0, most other probes) need the filter for the waveforms.
4. Cut the spike trains into runs (:func:`cut_spikes`) and map them onto each run's
   reference clock, the clock of the trial times.

The legacy MATLAB pipeline did the same with ``binaryForSortAp.bin`` and
``runOffset<probe>.mat``; it took the first ``ProbeDepth/10`` channels, which only holds for
Neuropixels 1.0 bank 0.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ephyscon.session import Session, StreamKind

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording, BaseSorting

    from ephyscon.session_config import ProbeConfig, SessionConfig
    from ephyscon.sync import RunSync

logger = logging.getLogger(__name__)

DEFAULT_SORTER = "kilosort4"
# sorter parameters that differ from the sorters' defaults
DEFAULT_SORTER_PARAMS: dict[str, dict[str, Any]] = {
    "kilosort4": {"clear_cache": True},  # lower GPU memory peaks (small GPUs)
}
ANALYZER_BAND_HZ = (300.0, 6000.0)  # band-pass of the recording the waveforms are taken from
ANALYZER_SEED = 0  # fixed seeds: the same spikes and noise chunks every time (reproducible waveforms)
ANALYZER_EXTENSIONS: dict[str, dict] = {
    "random_spikes": {"max_spikes_per_unit": 500, "seed": ANALYZER_SEED},
    "noise_levels": {"random_slices_kwargs": {"seed": ANALYZER_SEED}},
    "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
    "templates": {},
    "unit_locations": {"method": "monopolar_triangulation"},
    "template_metrics": {},
    "quality_metrics": {"metric_names": ["num_spikes", "firing_rate", "presence_ratio", "snr", "isi_violation"]},
}


@dataclass
class RunSegment:
    """Position of one run in the concatenated recording of a probe."""

    run: str
    start_sample: int
    num_samples: int


def sorter_settings(
    config: SessionConfig | None, probe_id: str, sorter: str | None = None, params: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    """Sorter and parameters of a probe: ``sorter``/``params`` (command line) over the probe's
    ``sorting`` over the session's ``[sorting]`` over the defaults."""
    probe_config = config.probes.get(probe_id) if config else None
    configured = (probe_config.sorting if probe_config and probe_config.sorting else None) or (
        config.sorting if config else None
    )
    name = sorter or (configured.sorter if configured else DEFAULT_SORTER)
    merged = dict(DEFAULT_SORTER_PARAMS.get(name, {}))
    if configured is not None and configured.sorter == name:
        merged.update(configured.params)
    merged.update(params or {})
    return name, merged


def adapt_sorter_params(sorter: str, params: dict[str, Any], num_channels: int) -> dict[str, Any]:
    """Limit channel neighbourhoods to the number of channels (probes with few channels)."""
    params = dict(params)
    if sorter == "kilosort4":
        for key, default in (("nearest_chans", 10), ("nearest_templates", 100)):
            if params.get(key, default) > num_channels:
                logger.info("kilosort4: %s = %d (number of channels)", key, num_channels)
                params[key] = num_channels
    return params


def sorting_channels(recording: BaseRecording, probe_config: ProbeConfig | None) -> np.ndarray:
    """Channel ids used for sorting: contacts inside the brain, without bad channels."""
    channel_ids = np.asarray(recording.channel_ids)
    keep = np.ones(channel_ids.size, bool)
    if probe_config is not None:
        if probe_config.insertion_depth_um is not None:
            y = recording.get_channel_locations()[:, 1]  # distance from the tip
            keep &= y < probe_config.insertion_depth_um
        keep[[c for c in probe_config.bad_channels if c < keep.size]] = False
    return channel_ids[keep]


def concatenate_runs(
    session: Session, probe_id: str, run_names: list[str], probe_config: ProbeConfig | None = None
) -> tuple[BaseRecording, list[RunSegment]]:
    """AP streams of the given runs, restricted to the sorting channels, concatenated."""
    from spikeinterface.core import concatenate_recordings

    recordings, segments, start = [], [], 0
    channel_ids = None
    for name in run_names:
        recording = session.run(name).get_recording(StreamKind.AP, probe_id)
        if channel_ids is None:
            channel_ids = sorting_channels(recording, probe_config)
        recordings.append(recording.select_channels(channel_ids))
        n = recording.get_num_samples()
        segments.append(RunSegment(name, start, n))
        start += n
    return concatenate_recordings(recordings), segments


def cut_spikes(
    sorting: BaseSorting, segments: list[RunSegment], probe_id: str, syncs: dict[str, RunSync]
) -> dict[str, pd.DataFrame]:
    """Spikes of a concatenated sorting, split into runs.

    Returns ``{run: DataFrame(time, sample, unit_id)}`` with ``sample`` the probe sample within
    the run and ``time`` in seconds on the run's reference clock (that of the trial times).
    """
    spikes = sorting.to_spike_vector()
    samples, unit_ids = spikes["sample_index"], np.asarray(sorting.unit_ids)[spikes["unit_index"]]
    runs = {}
    for segment in segments:
        sel = (samples >= segment.start_sample) & (samples < segment.start_sample + segment.num_samples)
        local = samples[sel] - segment.start_sample
        runs[segment.run] = pd.DataFrame({
            "time": syncs[segment.run].to_reference_time(probe_id, local),
            "sample": local,
            "unit_id": unit_ids[sel],
        })
    return runs


def save_run_spikes(path: Path, spikes: pd.DataFrame) -> None:
    np.savez(path, **{c: spikes[c].to_numpy() for c in spikes.columns})


def load_run_spikes(path: Path) -> pd.DataFrame:
    with np.load(path) as data:
        return pd.DataFrame({k: data[k] for k in data.files})


def unit_table(analyzer: Any) -> pd.DataFrame:
    """One row per unit: the sorter's unit properties (e.g. Kilosort's ``KSLabel``), peak
    channel, location (um) and quality metrics."""
    sorting = analyzer.sorting
    table = pd.DataFrame(index=pd.Index(sorting.unit_ids, name="unit_id"))
    for prop in sorting.get_property_keys():
        values = sorting.get_property(prop)
        if prop != "original_cluster_id" and values.ndim == 1:
            table[prop] = values
    templates = analyzer.get_extension("templates").get_data(operator="average")
    peak = np.abs(templates).max(axis=1).argmax(axis=1)
    table["peak_channel"] = np.asarray(analyzer.channel_ids)[peak]
    locations = analyzer.get_extension("unit_locations").get_data()
    table["x_um"], table["y_um"] = locations[:, 0], locations[:, 1]
    for ext in ["quality_metrics", "template_metrics"]:
        if analyzer.has_extension(ext):
            table = table.join(analyzer.get_extension(ext).get_data())
    return table


def sort_probe(
    session: Session,
    probe_id: str,
    work_dir: Path,
    scratch_dir: Path,
    config: SessionConfig | None = None,
    syncs: dict[str, RunSync] | None = None,
    sorter_params: dict[str, Any] | None = None,
    n_jobs: int = 4,
    keep_binary: bool = False,
    sorter: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Sort one probe over all included runs; writes ``<work_dir>/sorting/<probe>/`` and the
    per-run spike tables ``<work_dir>/spikes/<run>_<probe>.npz``.

    ``scratch_dir`` holds the local binary copy (size: channels x samples x 2 bytes), deleted
    afterwards unless ``keep_binary``, and the Kilosort checkpoints (a few GB), deleted once
    Kilosort has finished. ``syncs`` maps run names to their
    :class:`RunSync`, needed to put spike times on the reference clock. ``sorter`` and
    ``sorter_params`` override the sorter of ``config`` (see :func:`sorter_settings`).
    """
    import spikeinterface.core as si
    from spikeinterface.sorters import run_sorter

    run_names = [
        r.name for r in session.runs
        if probe_id in r.probes and (config is None or (rc := config.run_config(r.name)) is None or rc.include)
    ]
    probe_config = config.probes.get(probe_id) if config else None
    recording, segments = concatenate_runs(session, probe_id, run_names, probe_config)
    sorter, params = sorter_settings(config, probe_id, sorter, sorter_params)
    params = adapt_sorter_params(sorter, params, recording.get_num_channels())
    out = work_dir / "sorting" / probe_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "segments.json").write_text(json.dumps({
        "runs": [asdict(s) for s in segments],
        "channel_ids": [str(c) for c in recording.channel_ids],
        "sampling_frequency": recording.sampling_frequency,
        "sorter": sorter,
        "sorter_params": params,
    }, indent=2, default=str), encoding="utf-8")
    logger.info(
        "%s: %d runs, %d channels, %.0f s", probe_id, len(segments), recording.get_num_channels(),
        recording.get_total_duration(),
    )

    binary_dir = scratch_dir / f"{session.name}_{probe_id}"
    checkpoint_dir = scratch_dir / f"{session.name}_{probe_id}_{sorter}_checkpoints"
    job = {"n_jobs": n_jobs, "chunk_duration": "1s", "progress_bar": True}
    if (binary_dir / "binary.json").exists() or (binary_dir / "si_folder.json").exists():
        local = si.load(binary_dir)
        logger.info("%s: reusing local copy %s", probe_id, binary_dir)
    else:
        logger.info("%s: writing local copy to %s", probe_id, binary_dir)
        local = recording.save(folder=binary_dir, format="binary", dtype="int16", overwrite=True, **job)

    sorter_dir = out / sorter
    if (sorter_dir / "spikeinterface_log.json").exists() and json.loads(
        (sorter_dir / "spikeinterface_log.json").read_text(encoding="utf-8")
    ).get("error") is False:
        from spikeinterface.sorters import read_sorter_folder

        sorting = read_sorter_folder(sorter_dir)
        logger.info("%s: reusing sorting in %s", probe_id, sorter_dir)
    else:
        logger.info("%s: running %s with %s", probe_id, sorter, params)
        if sorter == "kilosort4":
            from ephyscon.kilosort_checkpoints import checkpoints

            resumable: Any = checkpoints(checkpoint_dir)
        else:
            resumable = contextlib.nullcontext()
        with resumable:
            sorting = run_sorter(sorter, local, folder=sorter_dir, remove_existing_folder=True, verbose=True, **params)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    sorting = sorting.remove_empty_units()

    from spikeinterface.preprocessing import bandpass_filter

    analyzer_dir = out / "analyzer"
    filtered = bandpass_filter(local, freq_min=ANALYZER_BAND_HZ[0], freq_max=ANALYZER_BAND_HZ[1])
    analyzer = si.create_sorting_analyzer(
        sorting, filtered, format="binary_folder", folder=analyzer_dir, sparse=True, overwrite=True, job_kwargs=job
    )
    for name, kwargs in ANALYZER_EXTENSIONS.items():
        analyzer.compute(name, **kwargs, **(job if name in ("noise_levels", "waveforms") else {}))
    units = unit_table(analyzer)
    units.to_csv(out / "units.csv")

    spikes_dir = work_dir / "spikes"
    spikes_dir.mkdir(exist_ok=True)
    runs = cut_spikes(sorting, segments, probe_id, syncs or {}) if syncs else {}
    for run, table in runs.items():
        save_run_spikes(spikes_dir / f"{run}_{probe_id}.npz", table)
    logger.info(
        "%s: %d units (%d good), %d spikes", probe_id, len(units),
        int((units.get("KSLabel") == "good").sum()) if "KSLabel" in units else -1, sorting.count_total_num_spikes(),
    )
    if not keep_binary:
        # the recording, sorting and analyzer memory-map the binary; Windows can't delete it before they're gone
        del local, filtered, analyzer, sorting
        gc.collect()
        shutil.rmtree(binary_dir, ignore_errors=True)
        if binary_dir.exists():
            logger.warning("%s: could not delete the local copy %s", probe_id, binary_dir)
    return runs
