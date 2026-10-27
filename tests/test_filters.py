"""The reach of the acausal filters documented in docs/filters.md and in the output files."""

import numpy as np
import pytest
from scipy import signal
from spikeinterface.preprocessing._resampling_tools import get_polyphase_filter, get_resampling_factors

from ephysalis.lfp import LFP_FILTER_REACH_MS, LFP_RATE
from ephysalis.mua import MUA_FILTER_ORDER, MUA_HIGHPASS_HZ, MUA_HIGHPASS_REACH_MS


@pytest.mark.parametrize("rate", [30000.0, 30000.27, 25000.0, 2500.0])
def test_lfp_filter_reach(rate):
    """spikeinterface's resampling FIR is symmetric and reaches 10 ms, flat to 400 Hz, >= 60 dB above 700 Hz."""
    up, down, _ = get_resampling_factors(rate, LFP_RATE, 10000)
    taps, _ = get_polyphase_filter(rate, up, down, None)
    np.testing.assert_allclose(taps, taps[::-1])  # zero phase
    assert (len(taps) - 1) / 2 / (rate * up) * 1000 == pytest.approx(LFP_FILTER_REACH_MS, rel=1e-3)
    f, h = signal.freqz(taps, worN=100_000, fs=rate * up)
    gain = 20 * np.log10(np.abs(h) / np.abs(h[0]) + 1e-12)
    assert np.abs(gain[f <= 400]).max() < 0.05
    assert gain[(f >= 700) & (f <= rate * up / 2)].max() < -60


def test_mua_highpass_reach():
    """The filtfilt high-pass falls below 1e-3 of its peak within MUA_HIGHPASS_REACH_MS before an impulse."""
    fs = 30000.0
    sos = signal.butter(MUA_FILTER_ORDER, MUA_HIGHPASS_HZ, "highpass", fs=fs, output="sos")
    impulse = np.zeros(60001)
    impulse[30000] = 1
    response = np.abs(signal.sosfiltfilt(sos, impulse))
    before = response[:30000][::-1]  # output before the impulse, i.e. the reach into the future
    last = np.flatnonzero(before > 1e-3 * response.max()).max()
    assert last / fs * 1000 <= MUA_HIGHPASS_REACH_MS


def test_lfp_resampling_has_no_phase_shift():
    """Sines through the LFP resampling (spikeinterface resample) keep their phase and amplitude."""
    import spikeinterface.core as si
    import spikeinterface.preprocessing as spre

    fs, freqs = 30000.0, np.array([1.0, 10.0, 40.0, 100.0, 300.0])
    t = np.arange(int(2 * fs)) / fs
    sines = np.sin(2 * np.pi * freqs * t[:, None]).astype("float32")
    lfp = spre.resample(si.NumpyRecording(sines, fs), LFP_RATE, dtype="float32").get_traces()
    t_lfp = np.arange(lfp.shape[0]) / LFP_RATE
    inner = slice(100, -100)  # away from the edges
    expected = np.sin(2 * np.pi * freqs * t_lfp[:, None])
    np.testing.assert_allclose(lfp[inner], expected[inner], atol=3e-3)
