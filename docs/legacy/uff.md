# DataWave UFF (`.UFF`)

These are continuous recordings of DataWave Experimenter's Workbench, which ran on MS-DOS. They
are the 1999 recordings of monkey Eddi (Tallon-Baudry et al. 2004).

The format was reverse-engineered in the Tallon-Baudry BIDS conversion (`code/uff/uff.py`,
`code/bids/eddi.py`), and the reader consolidates that code. Its samples and pulse pairs equal
that code on 1999-02-12 and 1999-03-04. The P stream's start and period equal the Tallon-Baudry
DH5 files of blocks ED12022 and ED12023.

## Files

Two PCs acquired each recording block, 16 channels each:

- `M<stem>.UFF`: the M PC, whose clock is the reference;
- `P<stem>.UFF`: the P PC. The P file sometimes lacks its prefix (`<stem>.UFF`), and the M file
  sometimes lacks its extension.

A file is a UFF file if its first 16 bytes contain `UFF`. The header has no date, so a block's
start time is the M file's modification time minus its duration.

## Layout

A file header, which contains the acquisition script as text, is followed by records back to
back. The reader does not parse the header. It takes the first record to be the first place where
three plausible records follow each other.

Each record is a 122-byte header (61 little-endian int16 words) followed by the samples:

| Word (byte) | Content |
|---|---|
| 1 (2) | record marker `0x61` |
| 2 (4) | time stamp, uint16 in 0.1 ms ticks, wraps every 6.5536 s (not used) |
| 11 (22) | number of channels *n* |
| 12… (24…) | channel numbers, *n* words |
| 44 (88) | number of sample values in the record, over all channels |
| bytes 102–105 | float32 ADC scale, mV per count |

- **Plausible record:** word 1 is `0x61`, 0 < *n* ≤ 64, and word 44 is positive and a multiple of
  *n*.
- **Samples:** word 44 int16 values, channel-interleaved (`[sample][channel]`), from a 12-bit ADC.
  The record size is 122 + 2 × word 44.
- **End:** the data end at the first record that is not plausible. The bytes after it are counted
  as truncated.
- **Rate:** the sampling rate is 1 kHz per channel. It is not read from the file.

## Channels and events

- **Streams:** `M` (channels `M0`…`M15`) and `P` (`P0`…`P15`). Electrodes are defined in
  `session.toml` with `stream = "M"` or `"P"`.
- **Channel 15 of both PCs** carries the same event pulses: about 660 counts against a baseline of
  about 12, one pulse per stimulus event, threshold 330 counts. These pulses on M15 are TTL line 1.
- **Default aux roles:** `eye_x` = M13, `eye_y` = M14, `event` = line 1.

## P clock

The P clock runs about 0.4 % slow (0.996005 P samples per M sample) and has occasional steps. The
P samples are mapped onto the M clock in two steps:

1. **Pairing** (port of `eddi.pair_pulses`): the pulse onsets of M and P are paired, because a
   pulse may be missing or doubled on one PC.
   - The P position of every M pulse is predicted from the rate and offset of the last 400 pairs.
   - The nearest P pulse within 3 samples is taken.
2. **Fit** (port of `eddi.p_clock_regions`): *P = rate × M + offset_k*, with one common rate.
   - There is one offset per stretch between steps of the P clock.
   - A step is a jump of the pair residual by more than 2.5 P samples.

In DH5 output, P blocks get their true sample period and one INDEX region per stretch.

## Reader API

```{eval-rst}
.. autoclass:: ephysalis.legacy.uff.UffFile
   :members: traces

.. autofunction:: ephysalis.legacy.uff.pair_pulses

.. autofunction:: ephysalis.legacy.uff.p_clock_fit

.. autofunction:: ephysalis.legacy.uff.block_files
```
