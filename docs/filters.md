# Filters and their reach into the future

All filters of ephyscon that shape stored signals are **acausal**: zero phase, applied forward and
backward (`filtfilt`) or as a symmetric FIR. A sample at time *t* is therefore computed from input
before **and after** *t*. They introduce no delay and no phase shift (`filtfilt` cancels the phase of the forward pass; a symmetric FIR centred on the output sample has none: sines from 1 to 300 Hz keep their phase within 0.0001° through the LFP resampling), but an event at *t₀* leaks into the output
before *t₀*. Analyses that relate the signal (e.g. its phase) at one time to later events must
treat effects within the reach below as possibly caused by the future, and add the reach of their
own filters (a wavelet or Hilbert phase usually reaches much further than the filters here).

The reach is symmetric (as far into the past as into the future). For IIR filters applied with
`filtfilt` the impulse response is infinite; the reach is given where its magnitude falls below
10⁻³ (10⁻⁴) of its peak. `tests/test_filters.py` checks the numbers below.

| Signal | Filter | Reach into the future |
|---|---|---|
| raw (`export --raw`) | none (only the acquisition hardware's analog filters) | 0 |
| LFP (`ephyscon lfp`, 1 kHz) | spikeinterface `resample` (`scipy.signal.resample_poly` design): Kaiser-windowed FIR (β = 5), cutoff 500 Hz (−6 dB), flat to 400 Hz (< 0.02 dB), ≥ 62 dB above 700 Hz; applied as a symmetric (zero-phase) FIR. 601 taps at 30 kHz, 501 at 25 kHz, 101 (at 5 kHz after upsampling by 2) for the 2.5 kHz Neuropixels LF band | **10 ms** exactly (finite FIR, 10 output samples, for every input rate) |
| MUA (`ephyscon mua`, 1 kHz) | 4th-order Butterworth high-pass 300 Hz, `filtfilt`; rectified (no reach); then the LFP resampling filter | **≈ 13 ms**: high-pass 3 ms (6.5 ms) + resampling 10 ms |
| running speed (1 kHz) | derivative of a Gaussian (σ ≈ 42.5 ms, 1 s kernel, centred) on the 1 ms bin counts (port of `get_wheelspeed.m`) | 500 ms kernel; weights > 10⁻³ of the peak within **179 ms** |
| eye position (1 kHz) | mean over 1 ms bins, time stamp at the bin centre | 0.5 ms |
| photodiode (trial onsets) | three passes of a centred ±1 ms moving average before the threshold (port of `quickFilter_c6`) | 3 ms: a detected onset can precede the light step by up to 3 ms, depending on the step height relative to the threshold |
| waveforms and unit metrics (`ephyscon sort`, SortingAnalyzer) | 5th-order Butterworth band-pass 300–6000 Hz, `filtfilt` (spikeinterface `bandpass_filter`); not stored as a continuous signal | 6.1 ms (9.8 ms) |

Spike times come from the sorter (Kilosort4 by default), which filters the data itself before
detecting spikes; see its documentation.

The descriptions stored in the output files (NWB `filtering`, DH5 block comments, BIDS
`*_ecephys.json`) name the filters and their reach (`lfp.LFP_FILTERING`, `mua.MUA_FILTERING`).
