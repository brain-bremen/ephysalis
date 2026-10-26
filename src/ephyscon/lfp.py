"""Local field potential per run and probe, downsampled with an anti-alias filter.

Neuropixels 1.0 probes record a separate LF band (2.5 kHz, sample-locked to AP/12); 2.0 probes
only record a wideband AP stream, from which the LFP is derived. Both are resampled with
spikeinterface's polyphase resampler (which low-pass filters before downsampling; the legacy
MATLAB pipeline picked every 2.5th LF sample without a filter).

The resampling filter is acausal: a symmetric (zero-phase) Kaiser-windowed FIR that reaches
:data:`LFP_FILTER_REACH_MS` into the future (and the past) for every input rate; see
``docs/filters.md``.

LFP sample ``k`` lies at ``k / rate`` seconds on the probe's own clock, i.e. at AP sample
``k * 30000 / rate``; map it onto the reference clock with the run's ``RunSync``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ephyscon.session import Run, StreamKind

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

logger = logging.getLogger(__name__)

LFP_RATE = 1000.0  # Hz
# half length of spikeinterface's resample_poly FIR (10 * max(up, down) upsampled input samples),
# i.e. 10 output samples
LFP_FILTER_REACH_MS = 10.0
LFP_FILTERING = (f"polyphase resampling to {LFP_RATE / 1000:g} kHz (spikeinterface resample: Kaiser-windowed FIR, "
                 f"cutoff 500 Hz), zero phase (acausal, reaches {LFP_FILTER_REACH_MS:g} ms into the future)")


def lfp_recording(run: Run, probe_id: str, rate: float = LFP_RATE) -> BaseRecording:
    """Lazy LFP recording of one probe in one run (int16, same gains as the source stream)."""
    import spikeinterface.preprocessing as spre

    kind = StreamKind.LF if run.probes[probe_id].has_lf else StreamKind.AP
    source = run.get_recording(kind, probe_id)
    return spre.resample(source, rate, dtype=source.get_dtype())


def lfp_folder(work_dir: Path, run_name: str, probe_id: str) -> Path:
    return work_dir / "lfp" / f"{run_name}_{probe_id}"


def extract_lfp(
    run: Run, probe_id: str, work_dir: Path, rate: float = LFP_RATE, n_jobs: int = 4, overwrite: bool = False
) -> BaseRecording:
    """Write the LFP of one probe in one run to ``<work_dir>/lfp/<run>_<probe>/`` (a
    spikeinterface binary folder, readable with ``spikeinterface.load``) and return it."""
    import spikeinterface.core as si

    folder = lfp_folder(work_dir, run.name, probe_id)
    if (folder / "binary.json").exists() and not overwrite:
        return si.load(folder)
    recording = lfp_recording(run, probe_id, rate)
    logger.info("%s %s: writing LFP (%d channels, %.0f Hz) to %s", run.name, probe_id,
                recording.get_num_channels(), rate, folder)
    return recording.save(folder=folder, format="binary", overwrite=True, n_jobs=n_jobs,
                          chunk_duration="10s", progress_bar=False)
