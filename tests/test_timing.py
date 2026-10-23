"""Sync, photodiode trials and wheel speed (synthetic data)."""

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording

from ephyscon.auxsignals import (
    detect_photodiode_pulses,
    smooth_boxcar,
    threshold_crossings,
    wheel_speed,
)
from ephyscon.sync import ClockMap, EdgeAccumulator, Edges, binary_edges, digital_edges, match_edges


def _quick_filter_c6(x, n_per, f_order):
    """Literal port of the MATLAB MEX loop, as a reference for smooth_boxcar."""
    vec = np.asarray(x, float).copy()
    n = vec.size
    scale = 1 / (2 * n_per + 1)
    for _ in range(f_order):
        temp = np.cumsum(vec)
        out = vec.copy()
        out[:n_per] = vec[0]
        out[n - n_per :] = vec[n - 1]
        out[n_per] = temp[2 * n_per] * scale
        for xi in range(2 * n_per, n - 1):
            out[xi - n_per + 1] = (temp[xi + 1] - temp[xi - 2 * n_per]) * scale
        vec = out
    return vec


def test_smooth_boxcar_matches_matlab_mex():
    rng = np.random.default_rng(0)
    x = np.repeat(rng.integers(0, 5000, 40), 25).astype(float) + rng.normal(0, 50, 1000)
    np.testing.assert_allclose(smooth_boxcar(x, 7), _quick_filter_c6(x, 7, 3), rtol=0, atol=1e-6)


def test_threshold_crossings_fractional():
    x = np.array([0, 0, 10, 10, 10, 0, 0], float)
    rising, falling = threshold_crossings(x, 5)
    np.testing.assert_allclose(rising, [1.5])
    np.testing.assert_allclose(falling, [4.5])


def test_digital_edges_across_chunks():
    bit = 3
    level = np.zeros(1000, bool)
    level[99:300] = True  # rising at 99, falling at 300
    level[499:500] = True  # rising exactly at a chunk boundary (chunk = 100 samples)
    level[800:] = True
    word = (level.astype(np.int16) << bit) | 0b101  # other bits set as well
    recording = NumpyRecording(word[:, None], sampling_frequency=1000.0)
    edges = digital_edges(recording, 0, bit, chunk_s=0.1)
    np.testing.assert_array_equal(edges.samples, [99, 300, 499, 500, 800])
    np.testing.assert_array_equal(edges.rising, [True, False, True, False, True])
    assert not edges.initial_level
    reference = binary_edges(level, 1000.0)
    np.testing.assert_array_equal(edges.samples, reference.samples)


def _sync_edges(times_s, fs):
    samples = np.round(np.asarray(times_s) * fs).astype(np.int64)
    rising = np.arange(samples.size) % 2 == 0
    return Edges(samples, rising, fs)


def test_clock_map_recovers_offset_and_drift():
    # 1 Hz sync (edges every 0.5 s) recorded by a reference at 30 kHz and an aux stream at
    # nominally 25 kHz that started 3.2 ms later and runs 20 ppm fast
    true_edges = np.arange(1, 200) * 0.5
    ref = _sync_edges(true_edges, 30000.0)
    aux_rate = 25000.0 * (1 + 20e-6)
    aux = Edges(np.round((true_edges - 0.0032) * aux_rate).astype(np.int64), ref.rising.copy(), 25000.0)
    src, dst = match_edges(aux, ref)
    assert src.size == true_edges.size
    clock = ClockMap(src.astype(float), dst / ref.sampling_frequency, aux.sampling_frequency)
    assert clock.effective_sampling_frequency == pytest.approx(aux_rate, rel=1e-7)

    event_true = np.array([0.1, 12.3456, 99.7, 101.0])  # inside and outside the edges
    event_samples = (event_true - 0.0032) * aux_rate
    np.testing.assert_allclose(clock.to_reference_time(event_samples), event_true, atol=40e-6)
    np.testing.assert_allclose(clock.to_samples(clock.to_reference_time(event_samples)), event_samples, atol=1e-6)


def test_match_edges_ignores_extra_edges():
    true_edges = np.arange(1, 21) * 0.5
    ref = _sync_edges(true_edges, 30000.0)
    # the aux stream started later and misses the first 4 edges
    aux = _sync_edges(true_edges[4:] - 0.002, 25000.0)
    aux.rising[:] = ref.rising[4:]
    src, dst = match_edges(aux, ref)
    np.testing.assert_array_equal(dst, ref.samples[4:])


def test_photodiode_square_pulses_and_invalid_gating():
    fs = 30000.0
    x = np.zeros(int(3 * fs))
    onsets = [0.2, 1.0, 2.0]
    for t in onsets:
        x[int(t * fs) : int((t + 0.134) * fs)] = 8000  # threshold 4000 at half height
    invalid = np.zeros(x.size, bool)
    invalid[: int(0.5 * fs)] = True  # signal not valid yet: the first pulse is ignored
    pulses = detect_photodiode_pulses(x, fs, invalid)
    assert pulses.mode == "square"
    np.testing.assert_allclose(pulses.onsets / fs, onsets[1:], atol=1e-3)
    np.testing.assert_allclose((pulses.offsets - pulses.onsets) / fs, 0.134, atol=1e-4)


def test_photodiode_single_frame_pulses():
    fs = 30000.0
    x = np.full(int(3 * fs), 200.0)  # baseline above 100: no dropouts
    for t in (0.5, 1.5, 2.5):
        x[int(t * fs) : int((t + 0.016) * fs)] = 20000
    pulses = detect_photodiode_pulses(x, fs, n_trials=3)
    assert pulses.mode == "frame"
    np.testing.assert_allclose(pulses.onsets / fs, [0.5, 1.5, 2.5], atol=1e-3)
    assert np.isnan(pulses.offsets).all()


def test_wheel_speed_constant_running():
    # 450 pulses/s on a 900 ppr wheel with 8 cm radius: half a revolution per second
    edges = np.arange(0, 10, 1 / 450) + 1e-4
    speed = wheel_speed(edges, 10.0)
    assert speed.shape == (10000,)
    np.testing.assert_allclose(speed[1000:9000], np.pi * 8, rtol=0.01)


def test_edge_accumulator_uneven_chunks():
    rng = np.random.default_rng(1)
    level = np.repeat(rng.integers(0, 2, 60).astype(bool), rng.integers(1, 30, 60))
    word = level.astype(np.int16) << 6
    accumulator = EdgeAccumulator(6, 1000.0)
    bounds = [0, 1, 7, 50, 51, 300, level.size]
    for a, b in zip(bounds[:-1], bounds[1:]):
        accumulator.add(word[a:b])
    reference = binary_edges(level, 1000.0)
    edges = accumulator.edges()
    np.testing.assert_array_equal(edges.samples, reference.samples)
    np.testing.assert_array_equal(edges.rising, reference.rising)
    assert accumulator.num_samples == level.size
