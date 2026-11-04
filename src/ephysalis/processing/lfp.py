"""Local field potential per run and probe, downsampled with an anti-alias filter.

Neuropixels 1.0 probes record a separate LF band (2.5 kHz, sample-locked to AP/12); 2.0 probes
only record a wideband AP stream, from which the LFP is derived. Both are resampled with the
polyphase resampler of :mod:`ephysalis.processing.resampling` (which low-pass filters before
downsampling; the legacy MATLAB pipeline picked every 2.5th LF sample without a filter), with the
:class:`~ephysalis.step_settings.LfpSettings` of the step (``[steps.lfp]``). The resampling filter is
acausal: a symmetric (zero-phase) Kaiser-windowed FIR (:meth:`LfpSettings.filtering`).

LFP sample ``k`` lies at ``k / rate`` seconds on the probe's own clock, i.e. at AP sample
``k * 30000 / rate``; map it onto the reference clock with the run's ``RunSync``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ephysalis.recording_session.model import Run, StreamKind
from ephysalis.step_settings import LfpSettings, read_step_settings, write_step_settings

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

logger = logging.getLogger(__name__)


def lfp_recording(run: Run, probe_id: str, settings: LfpSettings) -> BaseRecording:
    """Lazy LFP recording of one probe in one run (int16, same gains as the source stream)."""
    from ephysalis.processing.resampling import resample

    kind = StreamKind.LF if run.probes[probe_id].has_lf else StreamKind.AP
    source = run.get_recording(kind, probe_id)
    return resample(source, settings, dtype=source.get_dtype())


def lfp_folder(work_dir: Path, run_name: str, probe_id: str) -> Path:
    return work_dir / "lfp" / f"{run_name}_{probe_id}"


def lfp_settings(work_dir: Path) -> LfpSettings:
    """The settings the LFP in ``work_dir`` was made with (``<work_dir>/lfp/settings.json``)."""
    return read_step_settings(work_dir / "lfp", LfpSettings)


def extract_lfp(run: Run, probe_id: str, work_dir: Path, settings: LfpSettings, overwrite: bool = False) -> BaseRecording:
    """Write the LFP of one probe in one run to ``<work_dir>/lfp/<run>_<probe>/`` (a
    spikeinterface binary folder, readable with ``spikeinterface.load``) and return it; the settings go to
    ``<work_dir>/lfp/settings.json``. An existing LFP is reused only if it was made with the same settings."""
    import spikeinterface.core as si

    folder = lfp_folder(work_dir, run.name, probe_id)
    same = ((work_dir / "lfp" / "settings.json").exists()
            and lfp_settings(work_dir).output_settings() == settings.output_settings())
    if (folder / "binary.json").exists() and same and not overwrite:
        return si.load(folder)
    write_step_settings(work_dir / "lfp", settings)
    recording = lfp_recording(run, probe_id, settings)
    logger.info("%s %s: writing LFP (%d channels, %.0f Hz) to %s", run.name, probe_id,
                recording.get_num_channels(), settings.rate_hz, folder)
    from ephysalis.filesystem import remove_folder

    remove_folder(folder)  # spikeinterface's own overwrite fails on some NFS servers (ephysalis.filesystem)
    return recording.save(folder=folder, format="binary", overwrite=True, n_jobs=settings.n_jobs,
                          chunk_duration="10s", progress_bar=False)
