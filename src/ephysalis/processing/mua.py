"""Continuous multi-unit activity (MUA) envelope per run and probe.

The AP (wideband) signal is high-pass filtered, rectified and resampled with the polyphase resampler of
:mod:`ephysalis.processing.resampling` (which low-pass filters before downsampling), with the
:class:`~ephysalis.step_settings.MuaSettings` of the step (``[steps.mua]``; defaults: 4th-order Butterworth at
300 Hz forward-backward, 1 kHz, as ``filtfilt``, ``abs`` and anti-aliased decimation of the old converter).
:meth:`MuaSettings.filtering` describes the filters and how far they reach into the future.

MUA sample ``k`` lies at ``k / rate`` seconds on the probe's own clock, like the LFP
(:mod:`ephysalis.processing.lfp`). The work dir holds float32 µV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ephysalis.recording_session.model import Run, StreamKind
from ephysalis.step_settings import MuaSettings, read_step_settings, write_step_settings

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

logger = logging.getLogger(__name__)


def mua_recording(run: Run, probe_id: str, settings: MuaSettings) -> BaseRecording:
    """Lazy MUA envelope of one probe in one run (float32, µV)."""
    import spikeinterface.preprocessing as spre

    from ephysalis.processing.resampling import resample

    source = run.get_recording(StreamKind.AP, probe_id)
    scaled = spre.scale_to_uV(source)
    highpass = spre.highpass_filter(scaled, freq_min=settings.highpass_hz, filter_order=settings.highpass_order,
                                    ftype=settings.highpass_type, direction=settings.highpass_direction,
                                    margin_ms=settings.highpass_margin_ms, dtype="float32")
    return resample(spre.rectify(highpass) if settings.rectify else highpass, settings, dtype="float32")


def mua_folder(work_dir: Path, run_name: str, probe_id: str) -> Path:
    return work_dir / "mua" / f"{run_name}_{probe_id}"


def mua_settings(work_dir: Path) -> MuaSettings:
    """The settings the MUA in ``work_dir`` was made with (``<work_dir>/mua/settings.json``)."""
    return read_step_settings(work_dir / "mua", MuaSettings)


def extract_mua(run: Run, probe_id: str, work_dir: Path, settings: MuaSettings, overwrite: bool = False) -> BaseRecording:
    """Write the MUA envelope of one probe in one run to ``<work_dir>/mua/<run>_<probe>/`` (a
    spikeinterface binary folder) and return it; the settings go to ``<work_dir>/mua/settings.json``. An
    existing MUA is reused only if it was made with the same settings."""
    import spikeinterface.core as si

    folder = mua_folder(work_dir, run.name, probe_id)
    same = ((work_dir / "mua" / "settings.json").exists()
            and mua_settings(work_dir).output_settings() == settings.output_settings())
    if (folder / "binary.json").exists() and same and not overwrite:
        return si.load(folder)
    write_step_settings(work_dir / "mua", settings)
    recording = mua_recording(run, probe_id, settings)
    logger.info("%s %s: writing MUA (%d channels, %.0f Hz) to %s", run.name, probe_id,
                recording.get_num_channels(), settings.rate_hz, folder)
    from ephysalis.filesystem import remove_folder

    remove_folder(folder)  # spikeinterface's own overwrite fails on some NFS servers (ephysalis.filesystem)
    return recording.save(folder=folder, format="binary", overwrite=True, n_jobs=settings.n_jobs,
                          chunk_duration="10s", progress_bar=False)
