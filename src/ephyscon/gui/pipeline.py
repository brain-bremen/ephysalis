"""Session pipeline steps for the GUI: the ``ephyscon`` command of each step, and a summary
of the outputs in the work dir. No Qt here, so it can be tested without a display."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# (step, label, description) in pipeline order
STEPS = [
    ("sync", "Sync", "extract the sync edges of all streams once (reads the AP files)"),
    ("timing", "Timing", "trial times of all paradigms, running speed and eye position"),
    ("calibrate", "Calibration", "channel gains from the calibration-sine run ([calibration] in session.toml)"),
    ("lfp", "LFP", "LFP of every run and probe at 1 kHz (mains removal with [line_noise])"),
    ("quality", "Channel selection", "good channels of the probes with [probes.<id>.quality] (CSD uses them)"),
    ("csd", "CSD", "current source density of the probes with [probes.<id>.csd]"),
    ("mua", "MUA", "MUA envelope of every run and probe at 1 kHz (300 Hz high-pass, rectified)"),
    ("sort", "Spike sorting", "spike sorting per probe over all runs, cut back into runs"),
    ("export", "Export", "one NWB or DH5 file, optionally in a BIDS dataset"),
]


@dataclass
class PipelineOptions:
    session: Path
    work_dir: Path
    config: Path | None = None
    steps: list[str] = field(default_factory=lambda: [s for s, _, _ in STEPS])
    n_jobs: int = 4
    scratch_dir: Path | None = None
    sorter: str | None = None  # None: from session.toml
    output_format: str = "nwb"
    bids_root: Path | None = None
    export_lfp: bool = True
    export_raw: bool = False


def step_commands(options: PipelineOptions) -> list[tuple[str, list[str]]]:
    """``[(step, argv)]`` in pipeline order; ``argv`` runs ``ephyscon`` in this Python."""
    common = ["--work-dir", str(options.work_dir)]
    if options.config is not None:
        common += ["--config", str(options.config)]
    commands = []
    for step, _, _ in STEPS:
        if step not in options.steps:
            continue
        args = [step, str(options.session), *common]
        if step in ("lfp", "csd", "mua", "sort"):
            args += ["--n-jobs", str(options.n_jobs)]
        if step == "sort":
            if options.scratch_dir is not None:
                args += ["--scratch-dir", str(options.scratch_dir)]
            if options.sorter:
                args += ["--sorter", options.sorter]
        if step == "export":
            args += ["--output-format", options.output_format]
            if options.bids_root is not None:
                args += ["--bids", str(options.bids_root)]
            if not options.export_lfp:
                args.append("--no-lfp")
            if options.export_raw:
                args.append("--raw")
        commands.append((step, [sys.executable, "-u", "-m", "ephyscon.cli.main", *args]))
    return commands


def work_dir_summary(work_dir: Path, session_name: str) -> list[tuple[str, str]]:
    """``[(output, description)]`` of what the steps have written to ``work_dir``."""
    rows: list[tuple[str, str]] = []
    if not work_dir.exists():
        return rows
    sync = sorted((work_dir / "sync").glob("*.npz"))
    if sync:
        rows.append(("sync", f"sync edges of {len(sync)} runs"))
    for csv in sorted((work_dir / "trials").glob("*.csv")):
        trials = pd.read_csv(csv)
        rows.append((f"trials/{csv.name}", f"{len(trials)} trials"))
    for kind, unit in (("running_speed", "running speed"), ("eye", "eye position")):
        files = sorted((work_dir / kind).glob("*.npy"))
        if files:
            rows.append((kind, f"{unit} of {len(files)} runs"))
    if (work_dir / "calibration.json").exists():
        rows.append(("calibration.json", "channel gains from the calibration run"))
    quality = sorted((work_dir / "quality").glob("*.json"))
    if quality:
        rows.append(("quality", f"channel selection of {len(quality)} run × probe recordings"))
    for kind in ("lfp", "csd", "mua"):
        folders = sorted(p for p in (work_dir / kind).glob("*") if (p / "binary.json").exists())
        if folders:
            rows.append((kind, f"{len(folders)} run × probe recordings"))
    events = sorted((work_dir / "events").glob("*.csv"))
    if events:
        rows.append(("events", f"{sum(len(pd.read_csv(p)) for p in events):,} events in {len(events)} runs"))
    for units_csv in sorted((work_dir / "sorting").glob("*/units.csv")):
        units = pd.read_csv(units_csv)
        good = f", {int((units.KSLabel == 'good').sum())} good" if "KSLabel" in units else ""
        rows.append((f"sorting/{units_csv.parent.name}", f"{len(units)} units{good}"))
    spikes = sorted((work_dir / "spikes").glob("*.npz"))
    if spikes:
        n = sum(int(np.load(p)["time"].size) for p in spikes)
        rows.append(("spikes", f"{n:,} spikes in {len(spikes)} run × probe tables"))
    for extension in (".nwb", ".dh5"):
        path = work_dir / f"{session_name}{extension}"
        if path.exists():
            rows.append((path.name, f"{path.stat().st_size / 1e9:.2f} GB"))
    return rows
