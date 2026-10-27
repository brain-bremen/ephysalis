# DAQ files (`.daq`)

The acquisition program of the Brain Research Institute, University of Bremen, wrote these files
from 1999 to 2006, with ADC64 and MC-128 cards. A session folder contains one `.daq` file per
recording block. File versions 1.0–3.3 exist.

The format was reconstructed from Michael Borisov's DAQ → HDF5 converter (V2.7). The reader was
checked on the Fritz and Mini recordings against the lab's BIDS conversions of them.

## Records

A file is a sequence of packed little-endian records:

```text
char   tag[4]    e.g. "VER1", "CR02"
int32  size      size of the record including these 8 bytes
...    body
```

The header records come first, in a fixed order, and the data records follow. Times are int64
nanoseconds of the DAQ clock. The clock need not restart at 0 in a new file.

| File version | Header records | Data records |
|---|---|---|
| 1.0, 1.1, 2.0 | `VER1 DATE NAME BORD FCR1` | `CR01`, `EV01`, `TD01` |
| 3.1, 3.2 | `VER1 DATE NAME BORD EDR1 CDR1` | `CR02`, `EV02`, `TD01` |
| 3.3 | `VER1 DATE NAME RECP BORD EDR1 CDR1` | `CR02`, `EV02`, `TD01` |

Version 3.0 files are corrupt and cannot be read.

### Header records

Offsets are from the start of the record, so the body starts at +8.

`VER1`
: +8 int16 program major, +10 program minor, +12 file major, +14 file minor version.

`DATE`
: +8 six int16 values: year, month, day, hour, minute, second. This is the start of the file in
  the local time of the acquisition PC, to 1 s. That clock was a few minutes off the stimulus PC's
  clock (see {doc}`vstim-td01`).

`NAME`
: +8 `char[40]` user, +48 `char[40]` subject. Both are NUL-terminated, Latin-1.

`RECP` (3.3 only)
: +8 uint16 recording type (0 continuous, 1 trial based), +10 int8 start trigger, +11 int8 stop
  trigger, +12 int32 *unknown*.

`BORD`
: +8 int32 number of boards *n*, then *n* × `char[20]` board names, e.g. `ADC64 F/H - Board 1` or
  `MC-128 - Board 1`.

`FCR1` (versions 1.x, 2.0)
: One board, all channels at one rate.
  - **Fields:** +8 float32 maximum voltage, +12 float32 minimum voltage, +16 int16 number of
    channels *n*, +18 int16 ADC bits, then from +20 *n* × (int16 channel number, int32
    amplification).
  - **Size:** 20 + 6*n*. In some 2000 files the size field holds only the low byte of this value.
  - **Sample rate:** not stored. It is the spacing of the `CR01` times.

`EDR1` (version 3)
: The nTrodes, i.e. groups of consecutive channels that share a sample rate. +8 int32 number *n*,
  then from +12 *n* × 12 bytes:
  - +0 int16 nTrode id;
  - +2 int8 type (1 = spike waveform snippets, otherwise continuous);
  - +3 int16 first channel;
  - +5 int8 number of channels;
  - +6 six bytes *unknown*.

`CDR1` (version 3)
: The channels. +8 int32 number *n*, then from +12 *n* × 22 bytes, all packed:

  | Offset | Type | Field |
  |---|---|---|
  | +0 | int16 | global channel number |
  | +2 | int16 | channel number on its board |
  | +4 | int16 | ADC bits |
  | +6 | float32 | maximum voltage |
  | +10 | float32 | minimum voltage |
  | +14 | float32 | amplification (0 = not stored) |
  | +18 | float32 | sample rate in Hz |

### Data records

| Record | Size | Content |
|---|---|---|
| `CR01` | 16 + 2·*n* | +8 int64 time, +16 one int16 sample of each of the *n* channels |
| `EV01` | 24 | +8 int64 time, +16 int32 ±line, +20 int32 line states (not used) |
| `CR02` | 18 + 2·*c*·*s* | +8 int64 time of the first sample, +16 uint16 nTrode id, +18 *s* samples × *c* channels of that nTrode, int16, `[sample][channel]` |
| `EV02` | 20 | +8 int64 time, +16 int32 ±line |
| `TD01` | 32 | +8 int64 time, +16 int32 trial number, +20 int32 stimulus number, +24 8 bytes *unknown* |

- **`EV01`/`EV02`:** the line is 1-based. It is positive for a rising edge and negative for a
  falling edge of that trigger line.
- **`TD01`:** written when the stimulus PC (VStim 1.x) starts a trial, with the numbers it
  received from it.
- **Spike nTrodes** also write `CR02` records, which hold waveform snippets. They are not read.

## Samples and gains

- **Counts:** samples are raw ADC counts in int16. 12-bit cards are left-aligned, so their values
  are multiples of 16.
- **Gain:** a count is (maximum − minimum voltage) / 65536, divided by the amplification where it
  is stored. This is the voltage at the ADC input, because the amplifier gains are not in the file.
- **Channel ids:** `AI<n>`, with *n* the global DAQ channel (0-based).

## Boards, streams and runs

- **Board of a channel:** if the channel counts in the board names add up to the number of
  channels, they give the split (ADC64 = channels 0–63, MC-128 = 64–191). Otherwise a new board
  starts where the on-board channel number restarts.
- **Streams:** one per board. A board with several sample rates gets one stream per rate, named
  `<board> <rate> Hz`.
- **Pauses:** a paused recording continues in the same file. A `CR` record that starts later than
  the end of the previous one of its nTrode starts a new run, named `<file>_part<k>`. Each part's
  start time is the header time plus its offset on the DAQ clock.
- **Version 1.x/2.0 data:** one `CR01` record per sample frame. Runs of consecutive `CR01`
  records can be found with a strided view of the file, which is faster than reading every record
  head.

## Clocks of two boards

In the 2006 setup an MC-128 (board 2) recorded next to the ADC64 (board 1). The boards share no
clock, and the MC-128 runs about 0.38 % fast. The MC-128 recorded copies of trigger lines 1–16
as its lines 17–32, on its own clock.

Matching each edge to its copy gives *board 2 time = offset + rate × board 1 time*, as
`synchro_timestamp_adj.m` of the lab did:
1. The start is the most frequent difference between the first 50 line-1 edges and their copies.
2. Every edge is then matched to its nearest copy and the line is refitted, 6 times.

Without copies, both boards are taken to share one clock.

## Damage seen in real files

- **`FCR1` size truncated** to its low byte (version 1.0, 2000): the reader recomputes it from the
  channel count.
- **Zero-filled end** of a preallocated file (version 1.x/2.0): the data end where only zero bytes
  remain.
- **Last record cut off** by the end of the file.
- **Foreign bytes inside a version 3 file:**
  - The reader skips forward to the next place where three valid records follow each other.
  - The first of these must have a time between −10 s and +600 s from the last good record.
  - The skipped ranges are listed in `run.extra["damaged"]`.
- **`CR02` record starting before the end of the previous one** (version 3.1): taken as a
  continuation, as the original converter did.

## Reader API

```{eval-rst}
.. autoclass:: ephysalis.legacy.daq.DaqFormat
   :members: read_trial_descriptors, read_ttl_events, clock_maps

.. autofunction:: ephysalis.legacy.daq.read_header

.. autofunction:: ephysalis.legacy.daq.read_index

.. autofunction:: ephysalis.legacy.daq.board_fit

.. autofunction:: ephysalis.legacy.daq.cache_dir
```
