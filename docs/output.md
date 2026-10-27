# Output files

`ephyscon export` writes one file per session from the work dir. Session time 0 is 1 s before
the first run, and the runs are placed by their start times. With `--per-run`, one file per run
is written, on the run's own clock.

## DH5 layout

| Block | Content |
|---|---|
| CONT 1 – 1600 | raw AP/wideband data (`export --raw`), one block per channel over all probes |
| CONT 1601 – 2000 | analog inputs of the aux stream (`export --raw`) |
| CONT 2001 – 3600 | LFP (`ChannelStatus` / `ChannelStatusDescription`: channel selection) |
| CONT 3601 | running speed; CONT 3602/3603 eye position x/y |
| CONT 4001 – 5600 | MUA envelope |
| CONT 8001 – 9600 | CSD (int16 scaled as `dh_csd`: max / 32766.99 A/m³ per count) |
| SPIKE `10000 · probe index + unit id` | spike times and mean waveform per unit |
| TRIALMAP | trials of all runs |
| EV02 | TTL edges (`+line`/`-line`), VStim network events (`word + 1000`), names as attributes (`EventCodes`/`EventNames`: `<role>_on` = active edge) |
| /Intervals | `<role>`: pulses of each trigger line with a role; `run_<name>`: the runs |
| /Markers | `trial_start` and the `[[events.markers]]` |

Streams whose clock drifts get extra INDEX regions. Streams more than 100 ppm off their nominal
rate get their true sample period. Trial parameters and unit metrics are in `<file>_trials.csv`
and `<file>_units.csv` next to the file.

## NWB

The NWB file contains:
- **Processing modules:**
  - `ecephys`: LFP and MUA per run and probe;
  - `csd`: the CSD;
  - `behavior`: running speed and eye position;
  - `events`: TTL events, `trigger_pulses`, `trigger_lines` and `markers`.
- **Acquisition:** the raw data, with `--raw`.
- **Trials and epochs:** trials with all their parameters as columns; runs as epochs, with
  `start_time_method`.
- **Units:** the units of the spike sorting.

The processing record is stored in `scratch/ephyscon_processing`.

## BIDS

`--bids <root>` puts the NWB files into a BIDS `ecephys` dataset (BEP032), with these sidecars:
- `channels.tsv` (with the channel selection) and `electrodes.tsv`;
- `events.tsv` (trials and trigger pulses);
- `scans.tsv` (`acq_time` and start-time method per file) and `sessions.tsv`.

Files are numbered `run-<n>` by their position among the included runs, and get `task-<label>`
from `session.toml`. DH5 files get a BIDS-like layout with the same sidecars.

Validate with the BEP032 schema:

```bash
uv run bids-validator-deno <root> --schema https://bids-specification.readthedocs.io/en/bep032/schema.json
```
