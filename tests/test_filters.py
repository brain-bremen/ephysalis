"""The filters of the LFP and MUA steps (their settings: ephysalis.step_settings) and how far they reach, as the
output files describe them."""

import numpy as np
import pytest
from scipy import signal
from spikeinterface.preprocessing._resampling_tools import get_polyphase_filter, get_resampling_factors

from ephysalis.processing.resampling import anti_alias_filter, highpass_reach_ms, resample
from ephysalis.step_settings import LfpSettings, MuaSettings

LFP = LfpSettings()


@pytest.mark.parametrize("rate", [30000.0, 30000.27, 25000.0, 2500.0])
def test_lfp_filter_reach(rate):
    """The default anti-alias FIR is spikeinterface's (SciPy's design), symmetric, reaches anti_alias_half_length
    output samples (10 ms), flat to 400 Hz, >= 60 dB above 700 Hz."""
    up, down, _ = get_resampling_factors(rate, LFP.rate_hz, LFP.max_denominator)
    taps, _ = anti_alias_filter(up, down, LFP.anti_alias_kaiser_beta, LFP.anti_alias_half_length, LFP.anti_alias_cutoff)
    np.testing.assert_allclose(taps, get_polyphase_filter(rate, up, down, None)[0])
    np.testing.assert_allclose(taps, taps[::-1])  # zero phase
    reach_ms = LFP.anti_alias_half_length / LFP.rate_hz * 1000
    assert (len(taps) - 1) / 2 / (rate * up) * 1000 == pytest.approx(reach_ms, rel=1e-3)
    assert f"reaches {reach_ms:g} ms into the future" in LFP.filtering()
    f, h = signal.freqz(taps, worN=100_000, fs=rate * up)
    gain = 20 * np.log10(np.abs(h) / np.abs(h[0]) + 1e-12)
    assert np.abs(gain[f <= 400]).max() < 0.05
    assert gain[(f >= 700) & (f <= rate * up / 2)].max() < -60


def test_mua_highpass_reach():
    """The filtfilt high-pass falls below 1e-3 of its peak within highpass_reach_ms before an impulse; a lower
    cutoff reaches further."""
    fs, mua = 30000.0, MuaSettings()
    sos = signal.butter(mua.highpass_order, mua.highpass_hz, "highpass", fs=fs, output="sos")
    impulse = np.zeros(60001)
    impulse[30000] = 1
    response = np.abs(signal.sosfiltfilt(sos, impulse))
    before = response[:30000][::-1]  # output before the impulse, i.e. the reach into the future
    last = np.flatnonzero(before > 1e-3 * response.max()).max()
    assert last / fs * 1000 <= highpass_reach_ms(mua) <= last / fs * 1000 + 0.1
    assert highpass_reach_ms(MuaSettings(highpass_hz=100)) > highpass_reach_ms(mua)
    assert f"about {highpass_reach_ms(mua):g} ms" in mua.filtering()


def test_lfp_resampling_has_no_phase_shift():
    """Sines through the LFP resampling keep their phase and amplitude; the default settings give exactly
    spikeinterface's resample."""
    import spikeinterface.core as si
    import spikeinterface.preprocessing as spre

    fs, freqs = 30000.0, np.array([1.0, 10.0, 40.0, 100.0, 300.0])
    t = np.arange(int(2 * fs)) / fs
    sines = np.sin(2 * np.pi * freqs * t[:, None]).astype("float32")
    lfp = resample(si.NumpyRecording(sines, fs), LFP, dtype="float32").get_traces()
    np.testing.assert_array_equal(lfp, spre.resample(si.NumpyRecording(sines, fs), LFP.rate_hz, dtype="float32").get_traces())
    t_lfp = np.arange(lfp.shape[0]) / LFP.rate_hz
    inner = slice(100, -100)  # away from the edges
    expected = np.sin(2 * np.pi * freqs * t_lfp[:, None])
    np.testing.assert_allclose(lfp[inner], expected[inner], atol=3e-3)
