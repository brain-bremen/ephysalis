"""BIDS microelectrode electrophysiology (``ecephys``) layout around the session NWB files.

One session NWB file becomes::

    <root>/dataset_description.json, README, participants.tsv/.json
    <root>/sub-<subject>/sub-<subject>_sessions.tsv
    <root>/sub-<subject>/ses-<session>/ecephys/sub-<subject>_ses-<session>_ecephys.nwb
                                              ..._ecephys.json   recording metadata
                                              ..._probes.tsv     one row per probe
                                              ..._electrodes.tsv one row per contact (NWB electrodes)
                                              ..._channels.tsv   one row per LFP channel
                                              ..._events.tsv     runs and trials (session time axis)

All sidecars are made from the NWB file, so they describe what is in it. A DH5 file
(:func:`add_dh5_to_bids`) gets the same layout with ``_ecephys.dh5``, its sidecars made from
the session metadata (:func:`ephyscon.nwb.session_metadata_nwbfile`) and the channels pointing
to its CONT blocks. BIDS only allows NWB and NIX data files, so such a dataset is BIDS-like
and the validator reports the ``.dh5`` files. The NWB files hold
LFP and sorted units, not the raw AP data; the dataset is still a raw dataset (``DatasetType =
"raw"``, ``GeneratedBy`` ephyscon), as in the BEP032 examples, because the derivative rules
of the BIDS schema are written for images (they require ``SpatialReference`` for every file).

``ecephys`` is not in a released BIDS version yet: the files follow BEP032 and validate with
``bids-validator-deno --schema https://bids-specification.readthedocs.io/en/bep032/schema.json``.

Labels are the subject and session
ids with everything but letters and digits removed; the session label also drops the subject
id prefix (``SOMxCD1_172_20250109_Recording1`` -> ``sub-SOMxCD1172``, ``ses-20250109Recording1``).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pynwb import NWBFile

logger = logging.getLogger(__name__)

BIDS_VERSION = "1.10.2-dev"  # version of the BEP032 schema
NA = "n/a"


def bids_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]", "", text)
    if not label:
        raise ValueError(f"no BIDS label can be made from {text!r}")
    return label


def bids_labels(subject_id: str, session_id: str) -> tuple[str, str]:
    """Subject and session label; the session label without the subject id prefix."""
    session = session_id.removeprefix(subject_id).lstrip("_-") or session_id
    return bids_label(subject_id), bids_label(session)


def ecephys_path(root: Path, subject: str, session: str, extension: str = ".nwb") -> Path:
    """Path of the session's data file in the BIDS dataset at ``root``."""
    return root / f"sub-{subject}" / f"ses-{session}" / "ecephys" / f"sub-{subject}_ses-{session}_ecephys{extension}"


def _na(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)) or (isinstance(value, str) and value == ""):
        return NA
    return value


def _write_tsv(table: pd.DataFrame, path: Path) -> None:
    table = table.replace("", None)
    table.to_csv(path, sep="\t", index=False, na_rep=NA, lineterminator="\n")


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _upsert_tsv(path: Path, row: dict, key: str) -> None:
    """Add ``row`` to the table at ``path``, replacing a row with the same ``key``."""
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()
    new = pd.DataFrame([{k: str(_na(v)) for k, v in row.items()}])
    if not table.empty:
        table = table[table[key] != row[key]]
    _write_tsv(pd.concat([table, new], ignore_index=True).sort_values(key), path)


def _lfp_series(nwbfile: NWBFile) -> list:
    if "ecephys" not in nwbfile.processing or "LFP" not in nwbfile.processing["ecephys"].data_interfaces:
        return []
    return list(nwbfile.processing["ecephys"]["LFP"].electrical_series.values())


def _electrode_names(electrodes: pd.DataFrame) -> pd.Series:
    """``<probe>_e<index within the probe>``, e.g. ``imec0_e084``."""
    index = electrodes.groupby("group_name").cumcount()
    return electrodes.group_name + "_e" + index.map("{:03d}".format)


def probes_table(nwbfile: NWBFile) -> pd.DataFrame:
    electrodes = nwbfile.electrodes.to_dataframe()
    rows = []
    for name, group in nwbfile.electrode_groups.items():
        device, model = group.device, group.device.model
        rows.append({
            "probe_name": name,
            "type": "silicon-probe",
            "manufacturer": model.manufacturer if model is not None and model.manufacturer != "unknown" else None,
            "model": (model.model_number or model.name) if model is not None else None,
            "device_serial_number": device.serial_number,
            "associated_brain_region": None if group.location == "unknown" else group.location,
            "electrode_count": int((electrodes.group_name == name).sum()),
        })
    return pd.DataFrame(rows)


def electrodes_table(nwbfile: NWBFile) -> pd.DataFrame:
    electrodes = nwbfile.electrodes.to_dataframe()
    table = pd.DataFrame({
        "name": _electrode_names(electrodes),
        "probe_name": electrodes.group_name,
        "x": electrodes.rel_x,
        "y": electrodes.rel_y,
        "z": NA,
        "location": electrodes.location.replace("unknown", NA),
    })
    for column in ("channel_name", "in_brain", "bad_channel"):
        if column in electrodes:
            table[column] = electrodes[column].to_numpy()
    return table.reset_index(drop=True)


def lfp_streams(nwbfile: NWBFile) -> dict[int, tuple[list[str], float]]:
    """``{electrode row: (paths of its LFP series in all runs, rate)}`` of an NWB file."""
    streams: dict[int, tuple[list[str], float]] = {}
    for series in _lfp_series(nwbfile):
        for row in series.electrodes.data[:]:
            streams.setdefault(int(row), ([], float(series.rate)))[0].append(f"processing/ecephys/LFP/{series.name}")
    return streams


def channels_table(nwbfile: NWBFile, streams: dict[int, tuple[list[str], float]] | None = None) -> pd.DataFrame:
    """One row per electrode that has LFP; ``stream_id`` lists where its data are (default: the
    NWB file's LFP series, see :func:`lfp_streams`)."""
    electrodes = nwbfile.electrodes.to_dataframe()
    names = _electrode_names(electrodes).to_numpy()
    streams = lfp_streams(nwbfile) if streams is None else streams
    rows = []
    for row in sorted(streams):
        rows.append({
            "name": names[row].replace("_e", "_lfp", 1),
            "electrode_name": names[row],
            "type": "LFP",
            "units": "V",
            "sampling_frequency": round(streams[row][1]),
            "stream_id": ",".join(streams[row][0]),
            "status": "bad" if bool(electrodes.bad_channel.iloc[row]) else "good",
        })
    columns = ["name", "electrode_name", "type", "units", "sampling_frequency", "stream_id", "status"]
    return pd.DataFrame(rows, columns=columns)


def events_table(nwbfile: NWBFile) -> pd.DataFrame:
    """Runs (``trial_type = run``) and trials (``trial_type`` = paradigm) on the session time axis."""
    tables = []
    if nwbfile.epochs is not None:
        epochs = nwbfile.epochs.to_dataframe()
        tables.append(pd.DataFrame({
            "onset": epochs.start_time, "duration": epochs.stop_time - epochs.start_time,
            "trial_type": "run", "run": [tags[0] for tags in epochs.tags],
        }))
    if nwbfile.trials is not None:
        trials = nwbfile.trials.to_dataframe()
        events = pd.DataFrame({
            "onset": trials.start_time, "duration": trials.stop_time - trials.start_time,
            "trial_type": trials.paradigm,
        })
        tables.append(pd.concat([events, trials.drop(columns=["start_time", "stop_time"])], axis=1))
    if not tables:
        return pd.DataFrame(columns=["onset", "duration", "trial_type"])
    table = pd.concat(tables, ignore_index=True).sort_values(["onset", "trial_type"], kind="stable")
    for column in table.columns[3:]:  # write whole numbers without ".0" (columns with NaN are float)
        values = table[column]
        if pd.api.types.is_float_dtype(values) and np.all(np.mod(values.dropna(), 1) == 0):
            table[column] = values.astype("Int64")
    return table.reset_index(drop=True)


def events_json(nwbfile: NWBFile, table: pd.DataFrame) -> dict:
    columns: dict[str, Any] = {
        "onset": {"Description": "start on the session time axis (NWB file times)", "Units": "s"},
        "duration": {"Units": "s"},
        "trial_type": {"Description": "'run' for a whole run, otherwise the paradigm of the trial"},
        "run": {"Description": "run name (NWB epoch tag)"},
    }
    if nwbfile.trials is not None:
        for column in nwbfile.trials.colnames:
            if column in table and column not in columns:
                columns[column] = {"Description": nwbfile.trials[column].description}
    return {k: v for k, v in columns.items() if k in table}


def ecephys_json(nwbfile: NWBFile, power_line_frequency: float | str,
                 lfp: tuple[float, str] | None = None) -> dict:
    """``lfp`` = (rate, filter description) of the LFP (default: from the NWB file's LFP)."""
    series = _lfp_series(nwbfile)
    if lfp is None and series:
        lfp = (float(series[0].rate), series[0].filtering)
    models = [g.device.model for g in nwbfile.electrode_groups.values() if g.device.model is not None]
    epochs = nwbfile.epochs.to_dataframe() if nwbfile.epochs is not None else pd.DataFrame()
    if lfp:
        sampling_frequency: float | str = round(lfp[0])
        filters: Any = {"LFP": {"Description": lfp[1]}}
    else:
        resolution = nwbfile.units.resolution if nwbfile.units is not None else -1.0
        sampling_frequency, filters = (round(1 / resolution) if resolution > 0 else NA), NA
    data: dict[str, Any] = {
        "PowerLineFrequency": power_line_frequency,
        "SamplingFrequency": sampling_frequency,
        "SoftwareFilters": filters,
        "InstitutionName": nwbfile.institution,
        "Manufacturer": ", ".join(sorted({m.manufacturer for m in models if m.manufacturer not in (None, "unknown")})) or None,
        "ManufacturersModelName": ", ".join(sorted({m.model_number or m.name for m in models})) or None,
        "DeviceSerialNumber": ", ".join(g.device.serial_number for g in nwbfile.electrode_groups.values()
                                        if g.device.serial_number) or None,
        "RecordingDuration": float(epochs.stop_time.max()) if not epochs.empty else None,
        "RecordingType": "discontinuous" if len(epochs) > 1 else "continuous",
        "BodyPart": "BRAIN",
        "SampleEnvironment": "in vivo",
        "TaskName": ", ".join(sorted(set(nwbfile.trials["paradigm"][:]))) if nwbfile.trials is not None else None,
        "TaskDescription": nwbfile.session_description,
    }
    return {k: v for k, v in data.items() if v is not None}


def _participant(nwbfile: NWBFile, subject: str) -> dict:
    s = nwbfile.subject
    return {
        "participant_id": f"sub-{subject}",
        "species": s.species if s else None,
        "sex": s.sex if s else None,
        "age": s.age if s else None,
        "strain": s.strain if s else None,
        "genotype": s.genotype if s else None,
    }


PARTICIPANTS_JSON = {
    "species": {"Description": "Latin binomial (NWB subject.species)"},
    "sex": {"Description": "M, F, U or O (NWB subject.sex)"},
    "age": {"Description": "ISO 8601 duration at the session (NWB subject.age)"},
    "strain": {"Description": "strain of the subject"},
    "genotype": {"Description": "genotype of the subject"},
}


def _write_dataset_files(root: Path, name: str | None) -> None:
    from ephyscon.version import get_version_from_pyproject

    description = root / "dataset_description.json"
    if not description.exists():
        _write_json({
            "Name": name or root.name,
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "raw",
            "GeneratedBy": [{"Name": "ephyscon", "Version": get_version_from_pyproject(),
                             "CodeURL": "https://github.com/brain-bremen/ephyscon"}],
        }, description)
    readme = root / "README.md"
    if not readme.exists() and not (root / "README").exists():
        readme.write_text(
            f"# {name or root.name}\n\nNeuropixels sessions preprocessed with ephyscon: one NWB file per session "
            "with LFP (1 kHz), Kilosort4 units, trials and running speed. Raw data are not included.\n",
            encoding="utf-8",
        )
    if not (root / "participants.json").exists():
        _write_json(PARTICIPANTS_JSON, root / "participants.json")


def write_bids_sidecars(nwb_path: Path, root: Path, subject: str, session: str,
                        power_line_frequency: float | str = 50.0, dataset_name: str | None = None) -> None:
    """Write the sidecar files of the NWB file at ``nwb_path`` and update the dataset-level files."""
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(nwb_path), "r") as io:
        write_sidecars(io.read(), nwb_path, root, subject, session, power_line_frequency, dataset_name)


def write_sidecars(nwbfile: NWBFile, data_path: Path, root: Path, subject: str, session: str,
                   power_line_frequency: float | str = 50.0, dataset_name: str | None = None,
                   streams: dict[int, tuple[list[str], float]] | None = None,
                   lfp: tuple[float, str] | None = None) -> None:
    """Write the sidecars of the data file at ``data_path``, described by ``nwbfile``;
    ``streams`` and ``lfp`` describe the LFP when it is not in ``nwbfile`` (see
    :func:`channels_table`, :func:`ecephys_json`)."""
    root.mkdir(parents=True, exist_ok=True)
    _write_dataset_files(root, dataset_name)
    prefix = data_path.parent / data_path.name[:data_path.name.rindex("_ecephys")]
    _write_json(ecephys_json(nwbfile, power_line_frequency, lfp), Path(f"{prefix}_ecephys.json"))
    _write_tsv(probes_table(nwbfile), Path(f"{prefix}_probes.tsv"))
    _write_tsv(electrodes_table(nwbfile), Path(f"{prefix}_electrodes.tsv"))
    _write_json({
        "channel_name": {"Description": "channel id in the AP stream of the recording"},
        "in_brain": {"Description": "contact closer to the probe tip than the insertion depth"},
        "bad_channel": {"Description": "excluded from spike sorting"},
        "x": {"Units": "um"}, "y": {"Units": "um", "Description": "distance from the probe tip"},
    }, Path(f"{prefix}_electrodes.json"))
    _write_tsv(channels_table(nwbfile, streams), Path(f"{prefix}_channels.tsv"))
    events = events_table(nwbfile)
    _write_tsv(events, Path(f"{prefix}_events.tsv"))
    _write_json(events_json(nwbfile, events), Path(f"{prefix}_events.json"))
    _upsert_tsv(root / "participants.tsv", _participant(nwbfile, subject), "participant_id")
    _upsert_tsv(root / f"sub-{subject}" / f"sub-{subject}_sessions.tsv",
                {"session_id": f"ses-{session}", "acq_time": nwbfile.session_start_time.isoformat()},
                "session_id")
    logger.info("wrote the BIDS sidecars of %s", data_path)


def add_nwb_to_bids(nwb_path: Path, root: Path, subject: str, session: str, **kwargs: Any) -> Path:
    """Copy an existing session NWB file into the dataset at ``root`` and write its sidecars."""
    target = ecephys_path(root, subject, session)
    if nwb_path.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.stem + ".partial.nwb")
        shutil.copyfile(nwb_path, tmp)
        tmp.replace(target)
    write_bids_sidecars(target, root, subject, session, **kwargs)
    return target


def add_dh5_to_bids(session: Any, work_dir: Path, dh5_path: Path, root: Path, subject: str, session_label: str,
                    config: Any = None, **kwargs: Any) -> Path:
    """Copy a session DH5 file (:mod:`ephyscon.dh5`) into a BIDS-like dataset at ``root``
    (``_ecephys.dh5``, with its ``_trials.csv``/``_units.csv``) and write its sidecars."""
    import h5py

    from ephyscon.dh5 import sidecar_paths
    from ephyscon.lfp import LFP_FILTERING
    from ephyscon.nwb import session_metadata_nwbfile

    target = ecephys_path(root, subject, session_label, ".dh5")
    target.parent.mkdir(parents=True, exist_ok=True)
    if dh5_path.resolve() != target.resolve():
        tmp = target.with_name(target.stem + ".partial.dh5")
        shutil.copyfile(dh5_path, tmp)
        tmp.replace(target)
        for kind, source in sidecar_paths(dh5_path).items():
            if source.exists():
                shutil.copyfile(source, sidecar_paths(target)[kind])
    streams: dict[int, tuple[list[str], float]] = {}
    lfp = None
    with h5py.File(target, "r") as file:
        for name, group in file.items():
            if name.startswith("CONT") and group.attrs.get("SignalType") == "LFP":
                row = int(group.attrs["Channels"]["GlobalChanNumber"][0])
                rate = 1e9 / float(group.attrs["SamplePeriod"])
                streams[row] = ([name], rate)
                lfp = (rate, LFP_FILTERING)
    nwbfile = session_metadata_nwbfile(session, work_dir, config)
    write_sidecars(nwbfile, target, root, subject, session_label, streams=streams, lfp=lfp, **kwargs)
    ignore = root / ".bidsignore"
    lines = ignore.read_text(encoding="utf-8").splitlines() if ignore.exists() else []
    for pattern in ("*_ecephys_trials.csv", "*_ecephys_units.csv"):
        if pattern not in lines:
            lines.append(pattern)
    ignore.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
