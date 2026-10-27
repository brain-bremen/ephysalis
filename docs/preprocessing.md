# Calibration, mains removal, channel selection, CSD

These steps port the lab's preprocessing of epidural and intracortical array recordings
(Borisov's MATLAB/C++ tools and the BIDS conversions of the Fritz and Mini datasets). They work
on any input format.

```toml
[calibration]
run = "15hz_after_2005_03_30"      # 100 µV sine on all inputs (or file = "../other/ephysalis/calibration.json")

[line_noise]
enabled = true                     # dh_remove_50hz with the recorded mains (1 kHz LFP)
subtract_reference_mean = true     # as the BIDS conversions (default false: as dh_remove_50hz)

[aux_channels.analog]
mains = 46

[probes.epidural.quality]          # channel selection of the BIDS conversions
neighbour_radius_um = 3100         # intracortical arrays: 2100

[probes.epidural.csd]              # dh_csd V3.1 [RBF]: grid 3 mm, RBF parameter 2.2758, 1 Ohm m
exclude_inputs = ["AI20"]
```

Run `ephysalis calibrate`, then `lfp` (with mains removal), `quality` and `csd`.

- **`calibrate`** fits the recorded calibration sine per channel (`<work>/calibration.json`). The
  gains are applied in every later step.
- **`lfp` with `[line_noise]`** removes the mains with the recorded mains signal, in place on the
  1 kHz LFP.
- **`quality`** marks channels as good or bad per run (clipping, power, correlation with the
  neighbours, line noise; `<work>/quality/<run>_<probe>.json`). The statuses go into DH5
  `ChannelStatus`, the NWB electrodes table and BIDS `channels.tsv`.
- **`csd`** uses the good channels of each run, so the CSD channels can differ between runs.

On Fritz 2005-03-30 these steps reproduce the BIDS dataset:
- the LFP after mains removal equals it sample by sample (with `subtract_reference_mean`);
- the channel selection equals its `channels.tsv`.
