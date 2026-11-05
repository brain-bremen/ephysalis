# Source cache

`ephysalis run` can let the steps read a local copy of each recording session folder instead of the folder
itself, and work in a local copy of its work dir. This helps when the recordings and the work dirs lie on a network
share: the steps read the raw data memory-mapped, from several workers and in several steps, and fetch each
recording about twice over the network; and the later steps read the outputs of the earlier ones (LFP, CSD, MUA)
back. A copy is read once, sequentially, which is also the fastest way to read over the network.

## Use

```
ephysalis run pipeline.toml --parallel 3 --source-cache local --source-cache-dir /data/ephysalis-cache --source-cache-max-gb 500
```

| Option | Meaning |
|---|---|
| `--source-cache direct\|local` | `direct` (default): read the folders where they are; `local`: read a copy |
| `--source-cache-dir PATH` | where the copies go (a local disk; required for `local`) |
| `--source-cache-max-gb N` | all copies together take at most N GB (optional) |

The same settings can be put into the batch file as `[source_cache]` (`mode`, `dir`, `max_size_gb`, `reserve_gb`;
see the [configuration reference](configuration.md)); the options override it. Usually the cache belongs to the
machine, not to the pipeline, so the options are the better place.

## What happens

- Before the first step of a session that has to run, the runner copies its folder into the cache; the steps read
  the copy. A session with nothing left to do is not copied.
- While sessions compute, the runner **prefetches** the next sessions in order: at most `--parallel` copies
  ahead, each only when it fits. A session that starts while its prefetch runs waits for that copy.
- The copies together, with the local work dirs, stay below `max_size_gb` and leave `reserve_gb` (default 10 GB)
  free on the disk. A session
  waits until there is room; for a running session, prefetched copies of later sessions make room first. A session
  that alone is bigger than `max_size_gb` is read directly.
- A copy is deleted when its session ends. Copies of paused or interrupted sessions, and prefetched copies no
  session used, are kept: the next run uses them if the source files are unchanged (same sizes and times), or
  deletes them when their space is needed.
- The **work dir** stays local until the session ends: when a step has to run, the runner copies the session's
  work dir into `<cache>/.work/<recording_session>/` (only the files that differ), the steps work there, and when
  the session ends (done, failed, paused or interrupted) the changed files are copied back and the local copy is
  deleted. The export makes its files there too and copies each into the dataset in one go. The step records
  (`runner/`) are copied back last, so a copy that stops half-way leaves no record of a step whose outputs were
  not all copied: that step runs again. The session's lock stays in its work dir. Until the session ends,
  `ephysalis status` shows the work dir as it was when the session started.
- If a runner dies (crash, power), its local work dir is deleted by the next run; the work dir still has the state
  before the session started, and the steps since then run again. If copying back fails, the session fails and the
  results stay in the local copy until the session runs again.
- The results do not depend on the cache: the mode is no input of the steps (nothing runs again when it changes),
  and the provenance records the original folder (`recording_session`) and the copy (`read_from`), and the
  work dir's paths, with the local copy as `worked_in`.

The cache folder holds `<recording_session>/` (a copy), `<recording_session>.json` (its source, size, files and
owner), `.work/<recording_session>/` (a local work dir) and `.lock`. The work dirs grow while their steps run
(by a few GB per session): leave room for them in `max_size_gb` and `reserve_gb`. It must not
overlap a session folder or a work dir. It is safe to delete while no runner uses
it.
