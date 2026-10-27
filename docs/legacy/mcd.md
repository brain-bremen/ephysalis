# Multi Channel Systems MC_Rack (`.mcd`)

These files were written by MC_Rack 3.8 and 4.0 with a USB-ME256 system. They are the 2010–2011
recordings of monkey Iris. Multi Channel Systems never published the format, and its only
official reader is the 32-bit Windows Neuroshare library `nsMCDLibrary.dll` (version 3.1).

The layout below was reverse-engineered from the files, with that library as reference. On
8 files with header versions 10 and 11, everything the reader returns equals what the library
returns:
- all analog samples (electrode, analog-input and filter streams);
- sample rates, resolutions, hardware ids and labels;
- every digital and trigger event.

`tools/mcd_neuroshare_reference.py` dumps the library's values for such a comparison. It must run
in a 32-bit Python on Windows.

:::{note}
The Neuroshare library can mix up files that have the same name in different folders. In the
comparison it returned the data of another folder's `Data.mcd`. Treat its results for files with
common names with suspicion.
:::

Only continuous recordings are read. MC_Rack's triggered (sweep) mode, average streams and spike
streams are not read.

## Chunks

The whole file is a sequence of chunks, little-endian:

```text
char   tag[8]    padded with spaces, e.g. "STRMHDR ", "elec0001"
int64  size      size of the body (without these 16 bytes); -1 for MCSSTRM
...    body
```

Top-level layout of a file (offsets from a real 16 s file):

```text
0        MCSSTRM   size -1: everything that follows
16         LISThdr     the header list
32           MCSHDR      file header
2368         LAYOUT      electrode layout
...          STRMHDR     stream header      } once per stream,
...          STRMFMT     stream format      } in pairs
434350     IDX1POS     int64: file position of IDX1MCS
434374     trig1001  digi0001  filt0001  elec0004     block group 1 (100 ms)
...        trig1001  digi0001  filt0001  elec0004     block group 2 ...
5702854    IDX1MCS     index header
5703173    IDXIDX      block group index
5703509    IDXSWEEP    sweeps
```

## Header list (`LISThdr `)

### `MCSHDR  ` (2320 bytes)

| Offset | Content |
|---|---|
| +0 | 6 × uint16 version, e.g. `10 0 0 4 0 0` (MC_Rack 4.0), `10 0 8 3 0 0` (3.8), `11 0 0 4 0 5` (2011). The first value is the header version; the 3rd/4th are probably the minor/major MC_Rack version |
| +12 | original path of the file, NUL-terminated, e.g. `D:\MC_Rack Data\20100723_testdownsamp0001.mcd` |
| +2208 | device name, `USB-ME256 ` |
| +2288 | probably the device serial number (`00007`) |

The other fields are *unknown*.

### `LAYOUT  `

The electrode layout, with one record of about 1368 bytes per electrode: label (`01A`), MEA
layout name (`Linear64`) and amplifier name (`FA64I/S`). The reader does not use it.

### `STRMHDR `: stream header

Offsets are from the start of the body:

| Offset | Type | Content |
|---|---|---|
| +0 | uint16 | *unknown* (0 analog/digital, 3 filter, 5 trigger) |
| +2 | char[] | stream type, NUL-terminated: `analog` (electrodes and analog inputs), `digital`, `filter`, `trigger`, `average` |
| +11 | char[] | display name, e.g. `Electrode Raw Data 4`, `Digital Data`, `Filtered Data 1` |
| +784 | uint32 | sample rate in mHz (25 000 000 = 25 kHz) |
| +788 | uint16 | number of channels *n* |
| +792 | char[8] | stream tag, the name of its data chunks: `elec0001`, `anlg0001`, `filt0001`, `digi0001`, `trig0001`, `mean0001` |
| +804 | | *n* channel records of (size − 804) / *n* bytes each: 1104 for analog and digital streams, 1112 for filter and trigger streams |

Fields of a channel record:
- +4 uint32: channel index within the stream;
- +8 uint32: hardware id. This is the input of the USB-ME, and Neuroshare lists the channels by it.
- +790: label, NUL-terminated (`01A`, `61D`, `A1`, `D1`).

The other fields are *unknown*.

The tag prefix gives the stream's role:

| Prefix | Content |
|---|---|
| `elec` | electrode raw data, one stream per amplifier block |
| `anlg` | analog inputs of the USB-ME |
| `filt` | data filtered (and downsampled) online by MC_Rack |
| `digi` | the 16 digital inputs as one word per sample |
| `trig` | trigger events |
| `mean` | averages; not read |

### `STRMFMT `: stream format

Follows its `STRMHDR`. It is 24 bytes, followed for filter and trigger streams by a parameter
block:

| Offset | Type | Content |
|---|---|---|
| +2 | char | unit, `V` (0x20 for digital streams) |
| +4 | uint16 | bits per sample (16) |
| +6 | uint16 | AD zero, `0x8000` |
| +8 | float64 | resolution, unit per count (1.25e-7 V for electrodes, 1.25e-4 V for analog inputs, 0 for digital) |
| +16 | uint16 | 2 for analog streams, 1 for digital (*meaning unknown*) |
| +18 | uint16 | samples per block (2500 at 25 kHz = 100 ms; 100 for a 1 kHz filter stream) |
| +20 | uint16 | 40 in all files seen, probably the device's sample period in µs |
| +24 | | filter and trigger streams only: parameter block `PARAMFLT` (e.g. `Butterworth 2nd order`, cutoff frequency) or `PARAMTRG` (trigger settings, source stream) |

The resolution is the voltage at the amplifier input, because MC_Rack has already divided by the
amplifier gain.

## Data chunks

Data chunks are named by their stream's tag. They come in **block groups**: one chunk per stream
covering the same 100 ms, in a fixed order. The order is the same in all groups of a file, but it
is not necessarily the order of the stream headers, so a reader should take it from the first
group.

Every data chunk starts with the time of its block:

```text
uint64  start    ns since 1601-01-01 00:00, local time of the recording PC
uint64  end      ns, = start + 100 ms
...     payload
```

These times exceed the int64 range (2010 ≈ 1.29 × 10¹⁹ ns), so use unsigned arithmetic.

**Sample streams** (`elec`, `anlg`, `filt`, `digi`):
- **Payload:** `samples per block × n` uint16 values, `[sample][channel]`. Every chunk of a
  stream has the same size, 16 + 2 · samples · *n*.
- **Analog value:** sample − AD zero, as int16 counts. Times the resolution, this gives volts.
- **Digital stream:** one channel with the raw 16-bit word. Bit *k* is digital input *k*, and
  ephyscon reads it as TTL line *k* + 1.

**Trigger streams** (`trig`):
- **Payload:** the trigger events in the block, or nothing; a chunk without events has size 16.
- **Events:** packed records of 10, 14 or 22 bytes, all of the same size within a chunk:

  | Offset | Type | Content | In records of |
  |---|---|---|---|
  | +0 | uint64 | time, ns since 1601 (same clock as the block times) | 10, 14, 22 bytes |
  | +8 | uint16 | trigger value | 10, 14, 22 bytes |
  | +10 | uint16 | trigger number (manual triggers) | 14, 22 bytes |
  | +12 | 2 bytes | *unknown* | 14, 22 bytes |
  | +14 | int32 | trial number | 22 bytes |
  | +18 | int32 | stimulus number | 22 bytes |

- **22-byte records** are written with trial synchronisation, i.e. when VStim 1.x sends the trial
  and stimulus number through the USB box. There are 4 events per trial, 150 ms apart. ephyscon
  uses them as the trial descriptors of the run (see {doc}`vstim-td01`), like the `TD01` records
  of {doc}`daq`.

The digital inputs of a USB-ME that are not connected toggle constantly (about 2 million changes
in 16 s). Use `[events] lines_without_role = false` in `session.toml` and give the used lines a
role.

## Index

`IDX1POS ` (the first chunk after the header list) holds the file position of `IDX1MCS `.

`IDX1MCS `
: 303 bytes: uint16 1, then the original file path. Not used.

`IDXIDX  `
: Entries of `uint64 time, int64 file position` that point to the first chunk of a block group.
  - **Spacing:** not every group has an entry. The spacing *k* is 1 or 8 groups, and some files use
    other spacings.
  - **Positions between entries** follow from the fixed chunk sizes when the span holds no trigger
    events. Its size is then *m* × the size of a group with empty trigger chunks, where *m* is the
    time difference of the entries divided by 100 ms.
  - **Spans with trigger events** must be walked chunk by chunk.
  - **Block length:** read it from the first chunk's own start and end times. The spacing of the
    entries does not give it, because the index is sparse.
  - **Incomplete group:** a recording can stop inside its last block group, which then lacks the
    chunks of some streams. Drop that group.
  - **Speed:** with the index, a 2 GB file is indexed in 0.1–0.5 s, against 20–60 s for walking
    every chunk on a network share.

`IDXSWEEP`
: 16 bytes per sweep: uint64 start and end time. A continuous recording has one sweep, and more
  than one means a triggered recording.

## Split recordings

MC_Rack writes a long recording into several files of at most 2–3 GB: `name.mcd`, then
`name0001.mcd`, `name0002.mcd`, …. Note that `name0001.mcd` can also be a separate later
recording.

Each file has its own header, and its first block continues the previous file's last block
without a gap. ephyscon makes one run of consecutive files (sorted by name) that meet both
conditions:
- their streams (tags, channels, rates) are the same;
- the first block of each starts within half a block of the previous file's end.

For example, 2010-07-28 is 11 files that form one 59-minute run.

## Clock

All streams of a USB-ME share its sample clock. The block times place the blocks and the files
of a run. Channel offsets within a block are the sample index divided by the stream's rate.

## Neuroshare names

To compare with the library, note how it names a channel entity:
`<tag> <hardware id:4> <index:4> <label>`, e.g. `elec0001 0252 0000 61D`.
- **Analog values:** in the stream's unit (V), i.e. (sample − AD zero) × resolution.
- **Time stamps:** seconds from the start of the file, in steps of one sample (40 µs).
- **Digital inputs:** an analog entity of the whole word (`… D1`) plus one event entity per bit
  (`… D1 00` … `D1 15`). The events of a bit entity are its changes, `01` rising and `ff`
  falling.

## Reader API

```{eval-rst}
.. autoclass:: ephyscon.legacy.mcd.McdFormat
   :members: read_ttl_events, read_trigger_events, read_trial_descriptors, clock_maps

.. autoclass:: ephyscon.legacy.mcd.McdRecordingExtractor

.. autofunction:: ephyscon.legacy.mcd.parse_streams

.. autofunction:: ephyscon.legacy.mcd.scan_file

.. autofunction:: ephyscon.legacy.mcd.group_files

.. autofunction:: ephyscon.legacy.mcd.digital_transitions
```
