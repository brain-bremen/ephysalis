"""LFP extraction on the small real SpikeGLX dataset."""

import numpy as np
import pytest

from ephysalis.io.base import open_session
from ephysalis.lfp import extract_lfp, lfp_recording
from conftest import SPIKEGLX_CROP_SESSION

RUN = "3_rfmappingFine_g0"


@pytest.fixture(scope="module")
def run():
    return open_session(SPIKEGLX_CROP_SESSION).run(RUN)


@pytest.mark.parametrize("probe_id", ["imec0", "imec1"])
def test_lfp_matches_legacy(run, probe_id):
    """Same signal as the legacy LFP_1kHz (LF counts, every 2.5th sample without anti-alias
    filter); the residual difference is the aliasing of the legacy decimation."""
    lfp = lfp_recording(run, probe_id)
    assert lfp.sampling_frequency == 1000
    traces = lfp.get_traces().astype(float)
    legacy = np.load(run.path / "1_stimulusStageRF" / f"legacy_reference_{probe_id}.npz")["lfp_1kHz"]
    assert traces.shape == legacy.shape
    inner = slice(50, -50)
    for c in range(traces.shape[1]):
        assert np.corrcoef(traces[inner, c], legacy[inner, c])[0, 1] > 0.95
        assert np.std(traces[:, c]) == pytest.approx(np.std(legacy[:, c]), rel=0.05)


def test_extract_lfp_writes_folder(run, tmp_path):
    lfp = extract_lfp(run, "imec0", tmp_path, n_jobs=1)
    assert (tmp_path / "lfp" / f"{RUN}_imec0" / "binary.json").exists()
    assert lfp.get_num_samples() == 5000
    np.testing.assert_array_equal(lfp.get_traces(), lfp_recording(run, "imec0").get_traces())
