# `session.toml`

One file per session describes which runs are used, the probes, the sorter, the roles of the aux
channels and the events. It replaces the Excel font-colour convention, `Depths.txt`,
`BrokenChannelsProbe<N>_0Index.txt` and `folders2Combine.txt` of the MATLAB pipeline.
`ephysalis init` writes one from what the reader finds (and from the lab's Excel sheet).

```toml
schema_version = 6
session_id = "SOMxCD1_172_20250109_Recording1"
subject_id = "SOMxCD1_172"
task = "rfmapping"         # BIDS task entity (also per run)

[[runs]]
name = "3_rfmappingFine_g0"
include = true

[probes.imec0]
area = "V1"
insertion_depth_um = 2000.0
bad_channels = []          # 0-based

[sorting]
sorter = "kilosort4"

[aux_channels.analog]      # analog input per role
photodiode = 2
[aux_channels.digital]     # digital bit per role (bit = TTL line - 1)
wheel = 7

[events]
exclude_roles = ["sync", "wheel"]   # lines that switch too often to be events
lines_without_role = true           # false: only lines with a role (e.g. unconnected inputs toggle)

[events.lines.lever_pressed]   # pulses (DH5 /Intervals, NWB trigger_pulses, BIDS events.tsv)
active = "low"
description = "lever pressed"

[[events.markers]]             # DH5 /Markers (trial_start is always written)
name = "derived_sample_onset"
role = "interval2"
offset_ms = 650
```

## Probes without metadata

Readers detect Neuropixels probes. Other electrodes are defined as a list of channels of a stream:

```toml
[probes.v1]
stream = "elec0001"                # stream name (ephysalis inspect lists the streams)
channels = ["01A", "02A", "03A"]   # default: all channels of the stream
```

## Schema versions

`schema_version` is raised whenever the format changes. Older files are migrated when read, and
newer files are rejected. The changes per version are listed in the docstring of
`ephysalis.session_config`.
