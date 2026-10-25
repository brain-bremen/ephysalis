# ephyscon roadmap

ephyscon replaces the MATLAB preprocessing pipeline
(`Combine_SortScript_ChockaNpx2_KS4_mPC.m` →
`multiParadigmArtefactRemovalAndSpikesortingComb_contEye_CRSE.m`). It is a fork of
[oecon](https://github.com/brain-bremen/oecon) (Open Ephys → DH5 only), whose conversion
pipeline is still available as `ephyscon convert`.

## Decisions

- **Inputs**: pluggable readers (`ephyscon.input_formats` entry points) built on
  spikeinterface/neo. SpikeGLX and Open Ephys now, Multi Channel Systems later.
- **Stimulus systems**: pluggable (`ephyscon.stimulus_systems`). The current MATLAB
  "stimulusStage" system is one plugin. A VStim plugin (Open Ephys network messages,
  wrapping the existing `trialmap.py`) and future systems are added the same way.
- **Spike sorting**: spikeinterface + Kilosort4, sorted per probe over the concatenated runs.
- **Output**: one NWB file per session (all probes; paradigms as epochs plus a trials
  table). NWB first, DH5 second. BIDS microephys layout on top of the NWB files.
- **No backwards compatibility** with the legacy `mergedCombined_*.mat` files.
- **Later**: artefact removal, eye calibration (eye position is stored as uncalibrated volts), automatic curation.

## Phases

| # | Content | Status |
|---|---|---|
| 0 | Fork oecon, rename to ephyscon, plugin registries | done |
| 1 | Readers (SpikeGLX, Open Ephys), session model, `session.toml`, Excel migration, stimstage conditions | done |
| 2 | Sync (NIDQ ↔ imec sync edges), NIDQ decoding (photodiode trials, wheel, lasers), trial timing in stimulus plugins | done; open: splitting multi-paradigm runs by the paradigm line, laser calibration to mW |
| 3 | Sorting: concatenation, bad channels, NP1/NP2 geometry, KS4 via `run_sorter`, SortingAnalyzer, Phy round-trip; collect sync edges (`EdgeAccumulator`) in the same AP pass | done (`ephyscon sort`); open: Phy round-trip (curation), sync edges during the copy, multi-shank NP2 |
| 4 | LFP (NP1 `lf` stream, NP2 from AP, anti-aliased) and optional MUA, generalising `decimation.py`/`mua.py` to spikeinterface recordings | LFP done (`ephyscon lfp`); open: MUA |
| 5 | NWB writer, BIDS layout, extend the DH5 writer with spikes and trials | session NWB and DH5 writers and BIDS ecephys layout done (`ephyscon export`); open: raw AP in NWB (external link or none), probe geometry extension (ndx-probeinterface) |
| 6 | CLI/GUI batch processing for SpikeGLX sessions, re-running single steps (`rerun_steps_plan.md`), provenance | GUI session page (one session, steps as processes), provenance and `ephyscon reproduce`, config schema versions done; open: batch over many sessions |

## Legacy behaviour (phase 2)

From the MATLAB code, for 8-channel NIDQ rigs (defaults in `AuxChannelMap`):

| Signal | Channel | Notes |
|---|---|---|
| sync | digital bit 1 (NIDQ), bit 6 (imec SY) | MATLAB smooths edge differences and resamples NIDQ to 30 kHz; ephyscon maps edge times instead |
| photodiode | AI2 | trials: smoothed, gated by `photodiode_invalid`, threshold 4000 |
| photodiode invalid | digital bit 2 | high = photodiode signal not valid |
| paradigm | digital bit 3 | separates paradigms in old multi-paradigm runs |
| wheel | digital bit 7 | 900 pulses/rev, 8 cm radius (`get_wheelspeed.m`) |
| lasers red/green/blue | AI0 / AI4 / AI6 | calibration in `{Red,Green,Blue}_mW2BinVolt.mat` |

Validated against the MATLAB output on `tests/data/SOMxCD1_172_20250109_crop`:

- **Trial times** (`sampleIndsStart`/`Stop`) agree within 5 samples (0.17 ms). The rest is
  consistent with the MATLAB sync, which smooths the sync offset over 100 edges (~50 s) and
  so mixes in later clock drift near the start of a run; ephyscon interpolates between the
  neighbouring sync edges of each event.
- **Photodiode gating**: MATLAB zeroes the photodiode up to the first falling edge of the
  "invalid" line even if the line starts low (and fails if it never falls). ephyscon only
  gates when the line starts high.
- **Multi-paradigm runs** (several `<k>_stimulusStage*` folders per run, split by the paradigm
  line) are not supported yet. All 22 runs in `RawData` have one paradigm each.

`TrialTimeCorrectionInfo.mat` matches photodiode onsets to the write/creation times of
`trialWiseInformation/trialNr*.txt`. Creation times are lost when data is copied, so this
correction should be computed from data that is still on the acquisition machine, or
replaced by a more robust stimulus-side clock.

## Known issues in the legacy pipeline (not reproduced)

- Channels used for sorting were `1:ProbeDepth/10`. This assumes the Neuropixels 1.0 bank-0
  layout and is wrong for multi-shank 2.0 probes. ephyscon will select channels by contact
  depth from the probe geometry.
- Unit "depth" was the index of the peak channel. ephyscon uses unit locations in µm.
- For 2.0 probes, shanks were re-stacked into one virtual column for Kilosort.
- NP1 LFP was decimated from 2.5 kHz to 1 kHz by picking samples, without an anti-alias filter.
- **Running speed was about 8–9× too high** (8× over the full RF run of SOMxCD1_172, 9× in the 5 s crop). The wheel encoder line was resampled from the NIDQ
  rate to 30 kHz with MATLAB `resample` and thresholded at `> 0`; filter ringing turns every
  encoder pulse into several. Simulating that resampling reproduces the MATLAB
  `RunningVelocity_1kHz` (r = 0.98), so the rest of `get_wheelspeed.m` is ported correctly.
  ephyscon counts the real pulses. Analyses with running-speed thresholds must be re-tuned.
- The lab Excel sheets can disagree with the recordings. For example, the probe serial
  numbers of imec0 and imec1 are swapped in `SOMxCD1_172_20250109_Recording1`. The
  recording metadata is treated as ground truth.
