"""Current source density (CSD) of 2-D electrode arrays: port of Michael Borisov's ``dh_csd``
V3.1 [RBF] (C++, 2002-2006, Bremen Brain Research Institute; Gaussian RBF interpolation by Udo
Ernst; ``csd_array_generic.cpp``, ``rbf_gaussian.cpp``, ``gauss.c``), the CSD of the Fritz/Mini
epidural recordings (``ephysalis csd``, ``[probes.<id>.csd]`` in ``session.toml``).

Per sample, ``dh_csd`` fits ``z(p) = sum_j lambda_j g(|p - p_j|)``, ``g(d) = exp(-d^2 / (sigma^2 / 2))``
with ``sigma = grid spacing x RBF parameter``, exactly through the LFP of the input electrodes
(Gauss elimination of ``G lambda = z``, ``G_ij = g(|p_i - p_j|)``), takes the analytic Laplacian
``4 c exp(-c d^2) (c d^2 - 1)`` (``c = 2 / sigma^2``) at the output electrodes and returns
``-Laplacian / specific resistance``. All of it is linear, so here it is one matrix
``M = -L G^-1 / specific resistance`` (n_out x n_in); ``tests/test_csd.py`` checks it against a
literal port of the per-sample C++ code. Input in V and positions in m give A/m^3.

Inputs: the probe's channels without ``exclude_inputs`` and without calibration (NaN gain,
:mod:`ephysalis.calibration`); outputs: all channels of the probe. After channel selection
(``ephysalis quality``, :mod:`ephysalis.quality`) only the good channels of the run, as inputs
and outputs (as the Fritz/Mini BIDS conversions). The CSD is computed from the
LFP in the work dir (after mains removal, as in the lab: ``dh_remove_50hz`` before ``dh_csd``),
written to ``<work>/csd/<run>_<probe>/`` (float32, A/m^3), and stored in DH5 at CONT 8001...
as int16 with ``dh_csd``'s scaling: calibration = max |CSD| of the channel / (32767 - 0.01),
values rounded half away from zero.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment

if TYPE_CHECKING:
    from ephysalis.session import Run
    from ephysalis.session_config import CsdConfig

logger = logging.getLogger(__name__)

CSD_UNIT = "A/m^3"
DH5_FULL_SCALE = 32767.0 - 0.01  # dh_csd.cpp: ocalinfo = oamax / (32767.0 - 0.01)


def csd_operator(xy_in: np.ndarray, xy_out: np.ndarray, sigma: float, specific_resistance: float) -> np.ndarray:
    """CSD matrix (n_out x n_in): ``csd = M @ lfp_in``. Positions (n, 2) and ``sigma`` in m."""
    xy_in, xy_out = np.asarray(xy_in, float), np.asarray(xy_out, float)
    norm = 1.0 / (0.5 * sigma ** 2)  # rbf_gaussian.cpp
    gram = np.exp(-norm * ((xy_in[:, None] - xy_in[None]) ** 2).sum(-1))  # RbfPrepDist
    d_out = ((xy_out[:, None] - xy_in[None]) ** 2).sum(-1)
    laplacian = 4.0 * norm * np.exp(-norm * d_out) * (norm * d_out - 1)  # RbfPrepMulLapl
    return -(laplacian @ np.linalg.inv(gram)) / specific_resistance  # -Lapl(G^-1 z) / specres


def c_round(x: np.ndarray) -> np.ndarray:
    """C ``round``: to nearest, ties away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


class CsdRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, operator: np.ndarray, inputs: np.ndarray, input_gains_v: np.ndarray):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self._operator, self._inputs, self._gains = operator, inputs, input_gains_v

    def get_traces(self, start_frame, end_frame, channel_indices):
        lfp = self.parent_recording_segment.get_traces(start_frame, end_frame, self._inputs)
        csd = (lfp.astype(np.float64) * self._gains) @ self._operator.T
        out = csd.astype(np.float32)
        return out if channel_indices is None else out[:, channel_indices]


class CsdRecording(BasePreprocessor):
    """Lazy CSD (float32, A/m^3, gains 1) of an LFP recording at its channels ``outputs``
    (indices; default all)."""

    def __init__(self, recording: BaseRecording, operator: list | np.ndarray, inputs: list | np.ndarray,
                 outputs: list | np.ndarray | None = None):
        operator, inputs = np.asarray(operator, float), np.asarray(inputs, dtype=np.int64)
        outputs = np.arange(recording.get_num_channels()) if outputs is None else np.asarray(outputs, dtype=np.int64)
        BasePreprocessor.__init__(self, recording, channel_ids=recording.channel_ids[outputs], dtype="float32")
        gains_v = recording.get_channel_gains()[inputs] * 1e-6
        for segment in recording._recording_segments:
            self.add_recording_segment(CsdRecordingSegment(segment, operator, inputs, gains_v))
        self.set_channel_gains(1.0)
        self.set_channel_offsets(0.0)
        self.set_property("physical_unit", [CSD_UNIT] * self.get_num_channels())
        self._kwargs = {"recording": recording, "operator": operator.tolist(), "inputs": inputs.tolist(),
                        "outputs": outputs.tolist()}


def probe_operator(recording: BaseRecording, config: CsdConfig, good: set[str] | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(M, input channel indices, output channel indices)`` for the channels of an LFP
    recording with a probe; ``good``: the channels to use (default all)."""
    if not recording.has_probe():
        raise ValueError("the CSD needs the probe geometry (geometry in [probes.<id>] of session.toml)")
    xy = recording.get_channel_locations()[:, :2] * 1e-6  # µm -> m
    ids = [str(c) for c in recording.channel_ids]
    unknown = sorted(set(config.exclude_inputs) - set(ids))
    if unknown:
        raise ValueError(f"[probes.<id>.csd] exclude_inputs: no channels {unknown}")
    gains = recording.get_channel_gains()
    good = set(ids) if good is None else good
    inputs = np.array([i for i, c in enumerate(ids)
                       if c in good and c not in config.exclude_inputs and np.isfinite(gains[i])])
    outputs = np.array([i for i, c in enumerate(ids) if c in good])
    if inputs.size < 2:  # dh_csd: "Not enough channels available for CSD"
        raise ValueError(f"only {inputs.size} input channels for the CSD")
    operator = csd_operator(xy[inputs], xy[outputs], config.grid_spacing_m * config.rbf_parameter,
                            config.specific_resistance_ohm_m)
    return operator, inputs, outputs


def csd_folder(work_dir: Path, run_name: str, probe_id: str) -> Path:
    return work_dir / "csd" / f"{run_name}_{probe_id}"


def extract_csd(run: Run, probe_id: str, work_dir: Path, config: CsdConfig, n_jobs: int = 4,
                overwrite: bool = False) -> BaseRecording:
    """CSD of one probe in one run from its LFP in the work dir, to ``<work>/csd/<run>_<probe>/``."""
    import spikeinterface.core as si

    from ephysalis.lfp import lfp_folder

    folder = csd_folder(work_dir, run.name, probe_id)
    if (folder / "binary.json").exists() and not overwrite:
        return si.load(folder)
    source = lfp_folder(work_dir, run.name, probe_id)
    if not (source / "binary.json").exists():
        raise FileNotFoundError(f"no LFP in {source} (run `ephysalis lfp` first)")
    from ephysalis.quality import GOOD, load_quality

    lfp = si.load(source)
    quality = load_quality(work_dir, run.name, probe_id)
    good = None if quality is None else {c for c, r in quality.items() if r["status"] == GOOD}
    operator, inputs, outputs = probe_operator(lfp, config, good)
    csd = CsdRecording(lfp, operator, inputs, outputs)
    logger.info("%s %s: CSD of %d channels from %d inputs to %s", run.name, probe_id, csd.get_num_channels(),
                inputs.size, folder)
    saved = csd.save(folder=folder, format="binary", overwrite=True, n_jobs=n_jobs, chunk_duration="10s",
                     progress_bar=False)
    (folder / "csd.json").write_text(json.dumps({
        "method": "dh_csd V3.1 [RBF] (M. Borisov), ephysalis.csd", "unit": CSD_UNIT,
        "inputs": [str(lfp.channel_ids[i]) for i in inputs], "outputs": [str(lfp.channel_ids[i]) for i in outputs],
        "channel_selection": "good channels (ephysalis quality)" if good is not None else "none",
        **config.model_dump()}, indent=1), encoding="utf-8")
    return saved
