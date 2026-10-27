"""Processing record in the NWB/DH5 outputs and `ephysalis reproduce`, on the 5 s SpikeGLX crop."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import SPIKEGLX_CROP_SESSION
from ephysalis.cli.main import main
from ephysalis.provenance import (
    compare_environment,
    compare_inputs,
    environment,
    read_processing_record,
    record_config,
    replay_argv,
)
from ephysalis.session_config import SCHEMA_VERSION, load_session_config, save_session_config

RUN = "3_rfmappingFine_g0"
# DH5 export needs dh5io with SPIKE block writing (dh5io PR #15)
FORMATS = ["nwb", "dh5"] if hasattr(__import__("dh5io"), "Spike") else ["nwb"]


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """Work dir of timing, lfp and both exports with an edited session.toml."""
    work = tmp_path_factory.mktemp("work")
    config_path = work / "session.toml"
    assert main(["init", str(SPIKEGLX_CROP_SESSION), "-o", str(config_path)]) == 0
    config = load_session_config(config_path)
    config.probes["imec0"].area = "V1"
    config.probes["imec0"].bad_channels = [1]
    save_session_config(config, config_path)
    common = ["--work-dir", str(work), "--config", str(config_path)]
    assert main(["timing", str(SPIKEGLX_CROP_SESSION), *common]) == 0
    assert main(["lfp", str(SPIKEGLX_CROP_SESSION), *common, "--n-jobs", "1", "--rate", "500"]) == 0
    (work / "sorting" / "imec0").mkdir(parents=True)
    (work / "sorting" / "imec0" / "segments.json").write_text(json.dumps(
        {"runs": [], "sorter": "kilosort4", "sorter_params": {"clear_cache": True, "nearest_chans": 8}}))
    files = {}
    for fmt in FORMATS:
        files[fmt] = work / f"out.{fmt}"
        assert main(["export", str(SPIKEGLX_CROP_SESSION), *common, "--output-format", fmt,
                     "-o", str(files[fmt])]) == 0
    return work, config, files


@pytest.mark.parametrize("fmt", FORMATS)
def test_record_in_output(processed, fmt):
    work, config, files = processed
    record = read_processing_record(files[fmt])
    assert record is not None and record["record_schema_version"] == 1
    export = record["export"]
    assert export["config"]["schema_version"] == SCHEMA_VERSION
    assert record_config(record) == config
    assert record_config(record, "timing") == config and record_config(record, "lfp") == config
    assert export["argv"][:2] == ["ephysalis", "export"] and export["options"]["output_format"] == fmt
    assert record["steps"]["lfp"][-1]["options"]["rate"] == 500
    assert record["sorting"]["imec0"]["sorter_params"]["nearest_chans"] == 8
    assert export["environment"]["packages"]["numpy"] == np.__version__
    meta = [p for p in export["inputs"] if p.endswith(".ap.meta")]
    assert meta and all("sha256" in export["inputs"][p] for p in meta)
    assert not any(p.startswith(work.name) for p in export["inputs"])  # work dir is not an input


def test_compare():
    env = environment()
    assert env["packages"]["numpy"] == np.__version__
    env["ephysalis_source"] = {"commit": "abc", "dirty": False}
    assert compare_environment(env, env) == []
    changed = json.loads(json.dumps(env))
    changed["packages"]["numpy"] = "0.0.1"
    changed["ephysalis_source"] = {"commit": "def", "dirty": True}
    assert compare_environment(changed, env) == [
        f"numpy: 0.0.1 -> {np.__version__}", "ephysalis commit: def -> abc",
        "ephysalis had uncommitted changes when the output was made"]
    old = {"a.meta": {"size": 1, "sha256": "x"}, "b.bin": {"size": 5}}
    assert compare_inputs(old, {"a.meta": {"size": 1, "sha256": "y"}, "c": {"size": 2}}) == [
        "missing: b.bin", "new: c", "changed: a.meta"]


def test_replay_argv(processed, tmp_path):
    _, _, files = processed
    record = read_processing_record(files["nwb"])
    lfp = replay_argv(record["steps"]["lfp"][-1], SPIKEGLX_CROP_SESSION, tmp_path / "w", tmp_path / "c.toml")
    assert lfp[:2] == ["lfp", str(SPIKEGLX_CROP_SESSION)]
    assert lfp[lfp.index("--rate") + 1] == "500.0" and lfp[lfp.index("--n-jobs") + 1] == "1"
    assert lfp[lfp.index("--work-dir") + 1] == str(tmp_path / "w")
    export = replay_argv(record["export"], SPIKEGLX_CROP_SESSION, tmp_path / "w", None)
    assert "--output" not in export and "--config" not in export  # output goes to the new work dir


def test_reproduce(processed, tmp_path, capsys):
    work, config, files = processed
    new = tmp_path / "again"
    assert main(["reproduce", str(files[FORMATS[-1]]), "--work-dir", str(new)]) == 0  # dry run
    out = capsys.readouterr().out
    assert [line.split()[1] for line in out.splitlines() if line.startswith("ephysalis ")] == ["timing", "lfp", "export"]
    assert load_session_config(new / "session.toml") == config
    assert main(["reproduce", str(files[FORMATS[-1]]), "--work-dir", str(work)]) == 1  # same work dir

    assert main(["reproduce", str(files[FORMATS[-1]]), "--work-dir", str(new), "--run"]) == 0
    pd.testing.assert_frame_equal(pd.read_csv(new / "trials" / f"{RUN}_RF.csv"),
                                  pd.read_csv(work / "trials" / f"{RUN}_RF.csv"))
    lfp_files = sorted((work / "lfp").rglob("*.raw"))
    assert len(lfp_files) == 2  # imec0, imec1
    for original in lfp_files:
        assert (new / original.relative_to(work)).read_bytes() == original.read_bytes()
    assert (new / f"{SPIKEGLX_CROP_SESSION.name}.{FORMATS[-1]}").exists()


def test_records_before_the_rename(tmp_path):
    """Outputs, work dirs and records made under the old name (ephyscon) are still read."""
    import h5py

    from ephysalis.timing import default_work_dir

    record = {"packages": {"ephyscon": "0.3.0.dev0"}, "ephyscon_source": {"commit": "abc", "dirty": False}}
    with h5py.File(tmp_path / "old.nwb", "w") as f:
        f["scratch/ephyscon_processing/data"] = json.dumps(record)
    with h5py.File(tmp_path / "old.dh5", "w") as f:
        f.create_group("Operations/000_ephyscon_export").attrs["Processing"] = json.dumps(record)
    assert read_processing_record(tmp_path / "old.nwb") == record == read_processing_record(tmp_path / "old.dh5")
    current = {"packages": {"ephysalis": "0.3.0.dev0"}, "ephysalis_source": {"commit": "abc", "dirty": False},
               "python": None}
    assert compare_environment({**record, "python": None}, current) == []
    session = tmp_path / "session"
    assert default_work_dir(session) == session / "ephysalis"
    (session / "ephyscon").mkdir(parents=True)
    assert default_work_dir(session) == session / "ephyscon"
