"""Create a small real SpikeGLX test dataset by cropping a recorded session.

Writes valid SpikeGLX files (readable by SpikeGLX, neo, spikeinterface): every stream is cut
to ``--duration`` seconds from ``--start``, and imec streams keep only a subset of neural
channels plus the sync channel (a regular SpikeGLX saved-channel subset). Stimulus files of the
MATLAB "stimulusStage" system are copied in slimmed form, and reference values from the
legacy MATLAB output (``mergedCombined_Uncurated_DataForProbe<N>.mat``) are exported for
the same time window.

Example::

    uv run python tools/make_spikeglx_test_dataset.py \\
        X:/RawData/SOMxCD1_172/SOMxCD1_172_20250109_Recording1 3_rfmappingFine_g0 \\
        tests/data/SOMxCD1_172_20250109_crop --start 16.5 --duration 5 \
        --channels imec0=82-89 imec1=72-79
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

_META_RE = re.compile(r"^(?P<run>.+_g\d+)_t0\.(?P<stream>imec\d+\.(?:ap|lf)|nidq)\.meta$")
# small stimulus files that are copied unchanged
_STIM_FILES = [
    "params.mat",
    "TrialTimeCorrectionInfo.mat",
    "Red_mW2BinVolt.mat",
    "Green_mW2BinVolt.mat",
    "Blue_mW2BinVolt.mat",
    "red.mat",
    "green.mat",
    "blue.mat",
]
# datasets of stimulation.mat that are kept (plain arrays, no object references)
_STIMULATION_KEEP = ["screenInfo", "sessioninfo", "stimResults/stimulusOrder",
                     "stimResults/stimulusCount", "stimResults/repetition"]
_REDACT = ["sessioninfo/experimenter"]


def _read_meta_lines(path: Path) -> list[tuple[str, str]]:
    return [tuple(line.split("=", 1)) for line in path.read_text().splitlines() if "=" in line]  # type: ignore[misc]


def _tuples(value: str) -> list[str]:
    return re.findall(r"\(([^)]*)\)", value)


def parse_channels(specs: list[str]) -> dict[str | None, list[int]]:
    """Parse ``10 60``, ``82-89`` or ``imec0=82-89`` (per probe) into {probe or None: channels}."""
    result: dict[str | None, list[int]] = {}
    for spec in specs:
        probe, _, value = spec.rpartition("=")
        first, _, last = value.partition("-")
        channels = list(range(int(first), int(last or first) + 1))
        result.setdefault(probe or None, []).extend(channels)
    return result


def crop_stream(meta_path: Path, out_dir: Path, start_s: float, duration_s: float,
                channels: list[int] | None) -> None:
    lines = _read_meta_lines(meta_path)
    meta = dict(lines)
    n_saved = int(meta["nSavedChans"])
    rate = float(meta.get("imSampRate") or meta["niSampRate"])
    n_total = int(meta["fileSizeBytes"]) // (2 * n_saved)
    first = int(round(start_s * rate))
    n = min(int(round(duration_s * rate)), n_total - first)

    data = np.memmap(meta_path.with_suffix(".bin"), dtype=np.int16, mode="r",
                     shape=(n_total, n_saved))[first:first + n]
    replace: dict[str, str] = {}

    is_imec = meta.get("typeThis") == "imec"
    if is_imec and channels is not None:
        band = "lf" if meta_path.name.endswith(".lf.meta") else "ap"
        n_ap, n_lf, n_sy = (int(x) for x in meta["snsApLfSy"].split(","))
        n_neural = n_ap or n_lf
        columns = list(channels) + list(range(n_neural, n_saved))  # neural subset + sync
        data = data[:, columns]

        chan_map = _tuples(meta["~snsChanMap"])
        saved_entries = chan_map[1:]  # one entry per saved channel, in file order
        keep_entries = [saved_entries[c] for c in columns]
        acquired = [int(e.split(";")[1].split(":")[0]) for e in keep_entries]

        geom = _tuples(meta["~snsGeomMap"])
        replace.update(
            {
                "snsApLfSy": f"{len(channels)},0,{n_sy}" if band == "ap" else f"0,{len(channels)},{n_sy}",
                "snsSaveChanSubset": ",".join(map(str, acquired)),
                "~snsChanMap": "".join(f"({e})" for e in [chan_map[0], *keep_entries]),
                "~snsGeomMap": "".join(f"({e})" for e in [geom[0], *(geom[1 + c] for c in channels)]),
            }
        )

    data = np.ascontiguousarray(data)
    out_bin = out_dir / meta_path.with_suffix(".bin").name
    data.tofile(out_bin)
    replace.update(
        {
            "nSavedChans": str(data.shape[1]),
            "fileSizeBytes": str(data.nbytes),
            "fileTimeSecs": repr(n / rate),
            "firstSample": str(int(meta["firstSample"]) + first),
            "fileSHA1": hashlib.sha1(out_bin.read_bytes()).hexdigest().upper(),
            "userNotes": "",
        }
    )
    text = "\n".join(f"{k}={replace.get(k, v)}" for k, v in lines) + "\n"
    (out_dir / meta_path.name).write_text(text)
    print(f"  {meta_path.name}: {data.shape[1]} ch x {n} samples ({data.nbytes / 1e6:.1f} MB)")


def slim_stimulation_mat(src: Path, dst: Path) -> None:
    with h5py.File(src, "r") as fin, h5py.File(dst, "w", userblock_size=512) as fout:
        for path in _STIMULATION_KEEP:
            if path not in fin:
                continue
            parent = path.rsplit("/", 1)[0] if "/" in path else None
            if parent and parent not in fout:
                group = fout.create_group(parent)
                group.attrs.update(fin[parent].attrs)  # MATLAB_class and MATLAB_fields
                group.attrs.pop("MATLAB_fields", None)  # only a subset of fields is kept
            fin.copy(fin[path], fout[parent] if parent else fout, name=path.rsplit("/", 1)[-1])
        for path in _REDACT:
            if path in fout:
                attrs = dict(fout[path].attrs)
                del fout[path]
                ds = fout.create_dataset(path, data=np.array([[ord(c)] for c in "redacted"], dtype=np.uint16))
                ds.attrs.update(attrs)
    with open(src, "rb") as f:
        header = f.read(128)
    with open(dst, "r+b") as f:
        f.write(header)


def export_legacy_reference(paradigm_dir: Path, out_dir: Path, probes: dict[str, list[int]],
                            start_s: float, duration_s: float) -> None:
    for probe, channels in probes.items():
        legacy = paradigm_dir / f"mergedCombined_Uncurated_DataForProbe{probe.removeprefix('imec')}.mat"
        if not legacy.exists():
            continue
        with h5py.File(legacy, "r") as f:
            # trial on-/offsets in imec AP samples (30 kHz nominal), made relative to the crop
            offset = int(round(start_s * 30000))
            start = f["stims/sampleIndsStart"][()].ravel() - offset
            stop = f["stims/sampleIndsStop"][()].ravel() - offset
            in_window = (start >= 0) & (start < duration_s * 30000)
            lfp = slice(int(round(start_s * 1000)), int(round((start_s + duration_s) * 1000)))
            # Kilosort spikes (times in s) whose peak channel (1-based in the file) was kept
            times = f["SpikeTimes"][0] - start_s
            peak = f["SpikeDepths"][0].astype(int) - 1
            spikes = (times >= 0) & (times < duration_s) & np.isin(peak, channels)
            reference = {
                "trial_start_samples": start[in_window],
                "trial_stop_samples": stop[in_window],
                "running_velocity_1kHz": f["RunningVelocity_1kHz"][lfp, 0].astype(np.float32),
                "lfp_1kHz": f["LFP_1kHz"][lfp, channels].astype(np.float32),
                "lfp_channels": np.array(channels),
                "spike_times_s": times[spikes],
                "spike_unit_ids": f["SpikeIds"][0][spikes].astype(np.int32),
                "spike_peak_channels": peak[spikes],
            }
        np.savez_compressed(out_dir / f"legacy_reference_{probe}.npz", **reference)
        print(f"  legacy_reference_{probe}.npz: {len(reference['trial_start_samples'])} trials, "
              f"{len(reference['spike_times_s'])} spikes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("run")
    parser.add_argument("output", type=Path, help="Output session folder")
    parser.add_argument("--start", type=float, default=0.0, help="Start of the crop in s")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--channels", nargs="+", default=["10", "60", "110", "160"],
                        help="0-based neural channels to keep: 10 60, a range 82-89, "
                             "or per probe imec0=82-89")
    args = parser.parse_args()

    run_in = args.session / args.run
    run_out = args.output / args.run
    run_out.mkdir(parents=True, exist_ok=True)

    channel_specs = parse_channels(args.channels)
    print(f"Cropping {run_in} to {args.start}-{args.start + args.duration} s")
    probes: dict[str, list[int]] = {}
    for meta_path in sorted(run_in.glob("*.meta")):
        match = _META_RE.match(meta_path.name)
        if match is None:
            continue
        probe = match["stream"].split(".")[0]
        channels = channel_specs.get(probe, channel_specs.get(None))
        crop_stream(meta_path, run_out, args.start, args.duration, channels)
        if match["stream"].endswith(".ap") and channels is not None:
            probes[probe] = channels

    for paradigm_dir in sorted(run_in.glob("*_stimulusStage*")):
        out = run_out / paradigm_dir.name
        out.mkdir(exist_ok=True)
        for name in _STIM_FILES:
            if (paradigm_dir / name).exists():
                shutil.copy2(paradigm_dir / name, out / name)
        slim_stimulation_mat(paradigm_dir / "stimulation.mat", out / "stimulation.mat")
        export_legacy_reference(paradigm_dir, out, probes, args.start, args.duration)

    info = {
        "source_session": args.session.name,
        "run": args.run,
        "start_s": args.start,
        "duration_s": args.duration,
        "neural_channels": probes,
    }
    (args.output / "dataset.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
