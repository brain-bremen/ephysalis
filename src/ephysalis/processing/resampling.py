"""Polyphase resampling with a configurable anti-alias filter (``AntiAliasResampling`` settings of the LFP and MUA
steps), and the reach of the MUA high-pass into the future.

:class:`PolyphaseResampleRecording` is spikeinterface's ``resample`` (``scipy.signal.resample_poly`` per chunk, its
recording segments) with the FIR designed from the settings: ``firwin(2 * half_length * max(up, down) + 1,
cutoff / max(up, down), window=("kaiser", beta))``. The defaults (half length 10, beta 5, cutoff 1) are SciPy's
own design, i.e. the same as ``spikeinterface.preprocessing.resample``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor

if TYPE_CHECKING:
    from ephysalis.step_settings import AntiAliasResampling, MuaSettings


def anti_alias_filter(up: int, down: int, kaiser_beta: float, half_length: int, cutoff: float) -> tuple[np.ndarray, int]:
    """FIR coefficients for ``resample_poly`` and the number of input samples it reaches on each side."""
    from scipy.signal import firwin

    if up == down == 1:
        return np.ones(1), 0
    half = half_length * max(up, down)
    coefficients = firwin(2 * half + 1, cutoff / max(up, down), window=("kaiser", kaiser_beta))
    return coefficients, (half + up - 1) // up


class PolyphaseResampleRecording(BasePreprocessor):
    """A recording resampled to ``resample_rate`` with the anti-alias FIR of :func:`anti_alias_filter`."""

    def __init__(self, recording, resample_rate: float, kaiser_beta: float = 5.0, half_length: int = 10,
                 cutoff: float = 1.0, max_denominator: int = 10000, dtype=None):
        from spikeinterface.preprocessing._resampling_tools import get_resampling_factors
        from spikeinterface.preprocessing.filter import fix_dtype
        from spikeinterface.preprocessing.resample import ResampleRecordingSegment

        parent_rate = recording.get_sampling_frequency()
        up, down, achieved_rate = get_resampling_factors(parent_rate, resample_rate, max_denominator)
        dtype = fix_dtype(recording, dtype).str
        coefficients, margin = anti_alias_filter(up, down, kaiser_beta, half_length, cutoff)
        margin = ((margin + down - 1) // down) * down  # the padded chunk starts on the polyphase grid
        BasePreprocessor.__init__(self, recording, sampling_frequency=achieved_rate, dtype=dtype)
        for parent_segment in recording.segments:
            self.add_recording_segment(ResampleRecordingSegment(
                parent_segment, achieved_rate, parent_rate, margin, dtype, None, up, down, coefficients))
        self._kwargs = dict(recording=recording, resample_rate=resample_rate, kaiser_beta=kaiser_beta,
                            half_length=half_length, cutoff=cutoff, max_denominator=max_denominator, dtype=dtype)


def resample(recording, settings: AntiAliasResampling, dtype=None) -> PolyphaseResampleRecording:
    return PolyphaseResampleRecording(recording, settings.rate_hz, settings.anti_alias_kaiser_beta,
                                      settings.anti_alias_half_length, settings.anti_alias_cutoff,
                                      settings.max_denominator, dtype)


def highpass_reach_ms(settings: MuaSettings, sampling_frequency: float = 30000.0, threshold: float = 1e-3) -> float:
    """How far (ms) the forward-backward high-pass reaches into the future: the impulse response of the filter
    applied backward and forward falls below ``threshold`` of its peak before this (at ``sampling_frequency``),
    rounded up to 0.1 ms."""
    from scipy import signal

    sos = signal.iirfilter(settings.highpass_order, settings.highpass_hz, btype="highpass", ftype=settings.highpass_type,
                           fs=sampling_frequency, output="sos",
                           **({"rp": 0.1} if settings.highpass_type in ("cheby1", "ellip") else {}),
                           **({"rs": 60} if settings.highpass_type in ("cheby2", "ellip") else {}))
    n = int(sampling_frequency)  # 1 s
    impulse = np.zeros(2 * n + 1)
    impulse[n] = 1.0
    response = np.abs(signal.sosfiltfilt(sos, impulse))
    before = np.flatnonzero(response[:n] > threshold * response.max())
    samples = n - before[0] if before.size else 0
    return math.ceil(samples / sampling_frequency * 10000) / 10
