# SOMxCD1_172_20250109_crop

A small real SpikeGLX session for tests: 5 s (16.5–21.5 s) of run `3_rfmappingFine_g0` of
session `SOMxCD1_172_20250109_Recording1`. The recording has two Neuropixels 1.0 probes in
V1 and a NIDQ (PXIe-6341, 8 analog channels plus 1 digital word), with receptive-field
mapping by the MATLAB "stimulusStage" system.

Created with:

```bash
uv run python tools/make_spikeglx_test_dataset.py \
    <data>/SOMxCD1_172/SOMxCD1_172_20250109_Recording1 3_rfmappingFine_g0 \
    tests/data/SOMxCD1_172_20250109_crop --start 16.5 --duration 5 \
    --channels imec0=82-89 imec1=72-79
```

The channel blocks are the 8 adjacent channels per probe with the most Kilosort spikes in
this window (imec0: 1015 spikes from 20 units, imec1: 1169 spikes from 21 units).

| Content | Details |
|---|---|
| `*.imec{0,1}.ap.bin/.meta` | 8 neural channels + sync, 30 kHz, saved as a SpikeGLX channel subset; `firstSample` is shifted by the crop start |
| `*.imec{0,1}.lf.bin/.meta` | same channels, 2.5 kHz |
| `*.nidq.bin/.meta` | all 9 channels, 26041 Hz: photodiode AI2, lasers AI0/AI4/AI6, digital word with sync (bit 1), photodiode invalid (bit 2), paradigm (bit 3) and wheel encoder (bit 7) |
| `1_stimulusStageRF/` | `params.mat`, slimmed `stimulation.mat` (stimulus order, screen and session info; experimenter redacted), laser calibration, `TrialTimeCorrectionInfo.mat` |
| `1_stimulusStageRF/legacy_reference_imec{0,1}.npz` | output of the MATLAB pipeline for this window, relative to the crop start: trial on/offsets (imec samples), running velocity and LFP at 1 kHz, Kilosort spikes on the kept channels (`spike_times_s`, `spike_unit_ids`, `spike_peak_channels`) |

The window contains the first 23 trials of the paradigm; the first starts at 0.605 s.
