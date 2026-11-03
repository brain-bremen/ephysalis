# Installation and usage

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv tool install git+https://github.com/brain-bremen/ephysalis.git
uv tool upgrade ephysalis   # update
```

Spike sorting needs the `sorting` extra (Kilosort4 and torch) and an NVIDIA GPU.

## GUI

```bash
ephysalis-gui       # converter: sessions, session.toml, steps, outputs
ephysalis-inspect   # Open Ephys session inspector
```

## Steps

Each step is optional. Its outputs go into a work dir per session, and `export` writes the final
file from what is there. The steps are independent of the output format: only `export` makes
DH5 or NWB representations (e.g. the DH5 event codes, int16 scaling). The three extract steps
need the clock alignment; without `align-clocks` they extract the sync edges themselves. The
former `timing` step (all four) and `sync` (now `align-clocks`) still work but are deprecated.

| Step | Output |
|---|---|
| `align-clocks` | every stream of a run on the reference clock: the sync-pulse edges of all streams (cached in `sync/`, read by every later step) and the fit per stream (`clocks/<run>.json`: rate, offset, drift, residuals) |
| `extract-trials` | trial tables of all paradigms (stimulus system plugins) and better run start times from the stimulus system's clock |
| `extract-events` | edges of the digital lines and VStim network events (`events/<run>.csv`, independent of the output format) |
| `extract-behavior` | eye position and running speed at 1 kHz |
| `calibrate` | channel gains from a calibration-sine recording (`[calibration]`) |
| `lfp` | LFP per run and probe, 1 kHz; mains removal with the recorded mains (`[line_noise]`) |
| `quality` | good/bad channels per run from the LFP (`[probes.<id>.quality]`); the CSD then uses the good ones |
| `csd` | current source density of 2-D arrays (`[probes.<id>.csd]`, port of Borisov's `dh_csd`) |
| `mua` | MUA envelope per run and probe, 1 kHz (300 Hz high-pass, rectified) |
| `sort` | spike sorting per probe over all runs |
| `export` | one NWB or DH5 file per session (or per run, `--per-run`, on the run's own clock), optionally with the raw data and into a BIDS dataset |

All filters are zero phase (acausal): the LFP at time *t* depends on the raw data up to 10 ms after
*t*, the MUA up to about 13 ms (see {doc}`filters`).

## CLI

Every step takes one or more sessions. With several sessions, per-session paths contain
`{session}` (the session folder name); without `--work-dir` each session uses
`<session>/ephysalis`.

```bash
ephysalis inspect <session> --conditions             # runs, probes, streams, paradigms
ephysalis init <session> [--excel CR_Experiments25.xlsx]   # create session.toml

ephysalis align-clocks     <session> [<session> ...] --work-dir D:/work/{session}
ephysalis extract-trials   <session> [<session> ...] --work-dir D:/work/{session}
ephysalis extract-events   <session> [<session> ...] --work-dir D:/work/{session}
ephysalis extract-behavior <session> [<session> ...] --work-dir D:/work/{session}
ephysalis lfp     <session> [<session> ...] --work-dir D:/work/{session}
ephysalis mua     <session> [<session> ...] --work-dir D:/work/{session}
ephysalis sort    <session> [<session> ...] --work-dir D:/work/{session} --scratch-dir <fast local disk>
ephysalis export  <session> [<session> ...] --work-dir D:/work/{session} --output-format nwb|dh5 [--raw] [--bids <root>] [--per-run]

# redo the processing of an output file from its embedded processing record
ephysalis reproduce <file.nwb|file.dh5> --work-dir <new dir> [--run]
```

Every step records its invocation (arguments, `session.toml`, package versions, input file sizes
and checksums) in the work dir, and `export` embeds all of it in the output file, so
`ephysalis reproduce` can redo the processing.

## Library

```python
from ephysalis.io.base import open_session
from ephysalis.stimulus.base import detect_stimulus_system

session = open_session("/data/SOMxCD1_172_20250109_Recording1")
run = session.runs[0]
system = detect_stimulus_system(run)
conditions = system.load_conditions(system.discover(run)[0])  # pandas DataFrame, one row per trial
```
