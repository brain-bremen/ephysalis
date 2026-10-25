"""BIDS ecephys layout around the session NWB file of the 5 s SpikeGLX crop."""

import json

import pandas as pd
import pytest

from conftest import SPIKEGLX_CROP_SESSION
from ephyscon.bids import add_nwb_to_bids, bids_labels
from ephyscon.io.base import open_session
from ephyscon.nwb import SESSION_START_MARGIN_S, write_session_nwb
from ephyscon.session_config import ProbeConfig, SubjectConfig, default_session_config
from ephyscon.sync import synchronize_run
from ephyscon.timing import aux_channel_map
from test_nwb import RUN, work_dir  # noqa: F401 (fixture)

SUBJECT, SESSION = "SOMxCD1172", "20250109crop"


def test_bids_labels():
    assert bids_labels("SOMxCD1_172", "SOMxCD1_172_20250109_Recording1") == ("SOMxCD1172", "20250109Recording1")
    assert bids_labels("M1", "other-name") == ("M1", "othername")


@pytest.fixture(scope="module")
def bids_root(work_dir, tmp_path_factory):  # noqa: F811
    session = open_session(SPIKEGLX_CROP_SESSION)
    config = default_session_config(session)
    config.subject = SubjectConfig(species="Mus musculus", sex="F", age="P90D")
    config.probes["imec0"] = ProbeConfig(area="V1", insertion_depth_um=850, bad_channels=[1])
    run = session.run(RUN)
    syncs = {RUN: synchronize_run(run, aux_channel_map(run, config))}
    nwb = write_session_nwb(session, work_dir, work_dir / "bids_source.nwb", config, syncs)
    root = tmp_path_factory.mktemp("bids")
    add_nwb_to_bids(nwb, root, SUBJECT, SESSION)
    return root


def _tsv(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def _ecephys(root, suffix):
    return root / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "ecephys" / f"sub-{SUBJECT}_ses-{SESSION}_{suffix}"


def test_dataset_files(bids_root):
    description = json.loads((bids_root / "dataset_description.json").read_text(encoding="utf-8"))
    assert description["DatasetType"] == "raw" and description["GeneratedBy"][0]["Name"] == "ephyscon"
    participants = _tsv(bids_root / "participants.tsv")
    assert participants.to_dict("records") == [{"participant_id": f"sub-{SUBJECT}", "species": "Mus musculus",
                                                "sex": "F", "age": "P90D", "strain": "n/a", "genotype": "n/a"}]
    sessions = _tsv(bids_root / f"sub-{SUBJECT}" / f"sub-{SUBJECT}_sessions.tsv")
    assert list(sessions.session_id) == [f"ses-{SESSION}"]
    assert _ecephys(bids_root, "ecephys.nwb").exists()


def test_probes_electrodes_channels(bids_root):
    probes = _tsv(_ecephys(bids_root, "probes.tsv"))
    assert list(probes.columns[:2]) == ["probe_name", "type"]
    assert list(probes.probe_name) == ["imec0", "imec1"]
    assert list(probes.associated_brain_region) == ["V1", "n/a"]
    assert list(probes.manufacturer) == ["IMEC", "IMEC"]

    electrodes = _tsv(_ecephys(bids_root, "electrodes.tsv"))
    assert list(electrodes.columns[:5]) == ["name", "probe_name", "x", "y", "z"]
    assert len(electrodes) == 16 and electrodes.name.is_unique
    assert set(electrodes.probe_name) <= set(probes.probe_name)

    channels = _tsv(_ecephys(bids_root, "channels.tsv"))
    assert list(channels.columns[:5]) == ["name", "electrode_name", "type", "units", "sampling_frequency"]
    assert set(channels.electrode_name) <= set(electrodes.name)
    assert list(channels.status[channels.electrode_name.str.startswith("imec0")]) == ["good", "bad"] + ["good"] * 6
    assert set(channels.sampling_frequency) == {1000}
    assert channels.stream_id[0] == f"processing/ecephys/LFP/LFP_{RUN}_imec0"


def test_ecephys_json(bids_root):
    sidecar = json.loads(_ecephys(bids_root, "ecephys.json").read_text(encoding="utf-8"))
    assert sidecar["PowerLineFrequency"] == 50.0
    assert sidecar["SamplingFrequency"] == 1000
    assert sidecar["SoftwareFilters"]["LFP"]["Description"]
    assert sidecar["RecordingType"] == "continuous"
    assert sidecar["TaskName"] == "RF"


def test_events(bids_root, work_dir):  # noqa: F811
    events = _tsv(_ecephys(bids_root, "events.tsv"))
    assert list(events.columns[:3]) == ["onset", "duration", "trial_type"]
    assert list(events.trial_type).count("run") == 1 and list(events.trial_type).count("RF") == 23
    trials = pd.read_csv(work_dir / "trials" / f"{RUN}_RF.csv")
    rf = events[events.trial_type == "RF"]
    assert rf.onset.astype(float).to_numpy() == pytest.approx(trials.start_time + SESSION_START_MARGIN_S)
    columns = json.loads(_ecephys(bids_root, "events.json").read_text(encoding="utf-8"))
    assert set(columns) <= set(events.columns) and "xc" in columns


def test_rerun_replaces_rows(bids_root):
    add_nwb_to_bids(_ecephys(bids_root, "ecephys.nwb"), bids_root, SUBJECT, SESSION)
    assert len(_tsv(bids_root / "participants.tsv")) == 1
    assert len(_tsv(bids_root / f"sub-{SUBJECT}" / f"sub-{SUBJECT}_sessions.tsv")) == 1


BEP032_SCHEMA = "https://bids-specification.readthedocs.io/en/bep032/schema.json"


def test_bids_validator(bids_root):
    """No errors from the official validator (dev dependency ``bids-validator-deno``) with the
    BEP032 schema; needs network access to fetch the schema."""
    import shutil
    import subprocess

    validator = shutil.which("bids-validator-deno")
    if validator is None:
        pytest.skip("bids-validator-deno is not installed")
    result = subprocess.run([validator, str(bids_root), "--schema", BEP032_SCHEMA, "--json"],
                            capture_output=True, text=True, encoding="utf-8", timeout=300)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        pytest.skip(f"validator gave no report (offline?): {result.stderr[-500:]}")
    assert "ecephys" in report["summary"]["dataTypes"]
    errors = [(i["code"], i.get("location"), i.get("subCode")) for i in report["issues"]["issues"]
              if i["severity"] == "error"]
    assert not errors, errors
