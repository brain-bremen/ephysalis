"""Continuous multi-unit activity (MUA) envelope per run and probe.

The AP (wideband) signal is high-pass filtered (4th-order Butterworth at 300 Hz, forward and
backward, i.e. zero phase), rectified and resampled to 1 kHz with spikeinterface's polyphase
resampler, which low-pass filters before downsampling. This is the algorithm of
``ephyscon convert`` (``filtfilt`` high-pass, ``abs``, anti-aliased decimation).

Both filters are acausal (zero phase): the MUA at ``t`` depends on the input up to about
:data:`MUA_FILTER_REACH_MS` after ``t`` (high-pass 3 ms, resampling 10 ms); see
``docs/filters.md``.

MUA sample ``k`` lies at ``k / rate`` seconds on the probe's own clock, like the LFP
(:mod:`ephyscon.lfp`). The work dir holds float32 µV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ephyscon.lfp import LFP_FILTER_REACH_MS
from ephyscon.session import Run, StreamKind

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

logger = logging.getLogger(__name__)

MUA_RATE = 1000.0  # Hz
MUA_HIGHPASS_HZ = 300.0
MUA_FILTER_ORDER = 4
MUA_FILTER_MARGIN_MS = 100.0  # filter margin of each chunk; the 300 Hz high-pass settles within a few ms
# filtfilt high-pass: impulse response below 1e-3 of its peak after this (at 30 kHz)
MUA_HIGHPASS_REACH_MS = 3.0
MUA_FILTER_REACH_MS = MUA_HIGHPASS_REACH_MS + LFP_FILTER_REACH_MS
MUA_FILTERING = (f"{MUA_FILTER_ORDER}th-order Butterworth high-pass {MUA_HIGHPASS_HZ:g} Hz forward-backward, rectified, "
                 f"polyphase resampling to {MUA_RATE / 1000:g} kHz; zero phase (acausal, reaches about "
                 f"{MUA_FILTER_REACH_MS:g} ms into the future)")


def mua_recording(run: Run, probe_id: str, rate: float = MUA_RATE) -> BaseRecording:
    """Lazy MUA envelope of one probe in one run (float32, µV)."""
    import spikeinterface.preprocessing as spre

    source = run.get_recording(StreamKind.AP, probe_id)
    scaled = spre.scale_to_uV(source)
    highpass = spre.highpass_filter(scaled, freq_min=MUA_HIGHPASS_HZ, filter_order=MUA_FILTER_ORDER, ftype="butter",
                                    direction="forward-backward", margin_ms=MUA_FILTER_MARGIN_MS, dtype="float32")
    return spre.resample(spre.rectify(highpass), rate, dtype="float32")


def mua_folder(work_dir: Path, run_name: str, probe_id: str) -> Path:
    return work_dir / "mua" / f"{run_name}_{probe_id}"


def extract_mua(
    run: Run, probe_id: str, work_dir: Path, rate: float = MUA_RATE, n_jobs: int = 4, overwrite: bool = False
) -> BaseRecording:
    """Write the MUA envelope of one probe in one run to ``<work_dir>/mua/<run>_<probe>/`` (a
    spikeinterface binary folder) and return it."""
    import spikeinterface.core as si

    folder = mua_folder(work_dir, run.name, probe_id)
    if (folder / "binary.json").exists() and not overwrite:
        return si.load(folder)
    recording = mua_recording(run, probe_id, rate)
    logger.info("%s %s: writing MUA (%d channels, %.0f Hz) to %s", run.name, probe_id,
                recording.get_num_channels(), rate, folder)
    return recording.save(folder=folder, format="binary", overwrite=True, n_jobs=n_jobs,
                          chunk_duration="10s", progress_bar=False)
