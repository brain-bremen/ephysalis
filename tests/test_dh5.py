"""Session DH5 file from the work-dir outputs of the 5 s SpikeGLX crop (fixtures of test_nwb)."""

import numpy as np
import pandas as pd
import pytest
import spikeinterface.core as si

import dh5io
from conftest import SPIKEGLX_CROP_SESSION
from ephysalis.dh5 import LFP_FIRST_CONT_ID, RUNNING_SPEED_CONT_ID, lfp_regions, sidecar_paths, write_session_dh5
from ephysalis.io.base import open_session
from ephysalis.nwb import SESSION_START_MARGIN_S
from ephysalis.session_config import default_session_config
from ephysalis.sync import synchronize_run
from ephysalis.timing import aux_channel_map
from test_nwb import RUN, work_dir  # noqa: F401 (fixture)

if not hasattr(dh5io, "Spike"):
    pytest.skip("needs dh5io with SPIKE block writing", allow_module_level=True)


def test_lfp_regions_split_drifting_clocks():
    assert lfp_regions(1.0, 1000.0, 1000.0, 5000) == [(1_000_000_000, 0)]
    # 10 ppm fast: a new region every 0.1 / 1e-5 = 10000 samples
    regions = lfp_regions(0.0, 1000.01, 1000.0, 25000)
    assert [k for _, k in regions] == [0, 10000, 20000]
    assert regions[1][0] == round(10000 / 1000.01 * 1e9)


@pytest.fixture(scope="module")
def dh5_path(work_dir):  # noqa: F811
    session = open_session(SPIKEGLX_CROP_SESSION)
    config = default_session_config(session)
    run = session.run(RUN)
    syncs = {RUN: synchronize_run(run, aux_channel_map(run, config))}
    return write_session_dh5(session, work_dir, work_dir / "session.dh5", config, syncs)


def test_lfp_blocks(dh5_path, work_dir):  # noqa: F811
    with dh5io.DH5File(dh5_path, "r") as file:
        cont_ids = [c.id for c in file.get_cont_groups()]
        assert cont_ids == list(range(LFP_FIRST_CONT_ID, LFP_FIRST_CONT_ID + 16)) + [RUNNING_SPEED_CONT_ID]
        for probe, first in (("imec0", 0), ("imec1", 8)):
            source = si.load(work_dir / "lfp" / f"{RUN}_{probe}")
            block = file.get_cont_group_by_id(LFP_FIRST_CONT_ID + first + 1)
            np.testing.assert_array_equal(block.data[:, 0], source.get_traces(return_in_uV=False)[:, 1])
            assert block.calibration[0] == pytest.approx(source.get_channel_gains()[1] * 1e-6)
            assert block.sample_period == 1_000_000
        imec0 = file.get_cont_group_by_id(LFP_FIRST_CONT_ID).index
        imec1 = file.get_cont_group_by_id(LFP_FIRST_CONT_ID + 8).index
        assert len(imec0) == 1 and imec0["time"][0] > SESSION_START_MARGIN_S * 1e9 - 1e6
        # imec1 runs ~6.5 ppm fast: about 0.1 / 6.5e-6 = 15000 samples per region, so one region in 5 s
        assert imec0["time"][0] - imec1["time"][0] == pytest.approx(56_000, abs=10_000)


def test_spike_blocks_and_units_sidecar(dh5_path):
    units = pd.read_csv(sidecar_paths(dh5_path)["units"])
    assert list(units.spike_block) == [3, 7] and list(units.probe) == ["imec0", "imec0"]
    with dh5io.DH5File(dh5_path, "r") as file:
        assert sorted(s.id for s in file.get_spikes()) == [3, 7]
        unit3 = file.get_spike_by_id(3)
        assert not unit3.has_waveforms and unit3.name == "imec0 unit 3"
        np.testing.assert_array_equal(unit3.index, np.round((np.array([0.5, 2.0]) + SESSION_START_MARGIN_S) * 1e9))
        assert unit3.channels["BoardChanNo"][0] == 2  # AP84 is the third channel of the crop
        assert "kilosort4 unit 3" in unit3.comment and "label good" in unit3.comment


def test_trialmap_and_trials_sidecar(dh5_path, work_dir):  # noqa: F811
    trials = pd.read_csv(sidecar_paths(dh5_path)["trials"])
    csv = pd.read_csv(work_dir / "trials" / f"{RUN}_RF.csv")
    with dh5io.DH5File(dh5_path, "r") as file:
        trialmap = file.get_trialmap()
        assert len(trialmap) == 23
        np.testing.assert_array_equal(trialmap["StimNo"], csv.condition)
        np.testing.assert_allclose(trialmap["StartTime"] / 1e9, csv.start_time + SESSION_START_MARGIN_S, atol=1e-9)
        assert set(trialmap["Outcome"]) == {1}
    np.testing.assert_array_equal(trials.TrialNo, np.arange(23))
    np.testing.assert_array_equal(trials.xc, csv.xc)


def test_running_speed(dh5_path, work_dir):  # noqa: F811
    with dh5io.DH5File(dh5_path, "r") as file:
        block = file.get_cont_group_by_id(RUNNING_SPEED_CONT_ID)
        np.testing.assert_allclose(block.calibrated_data[:, 0], np.load(work_dir / "running_speed" / f"{RUN}.npy"),
                                   atol=block.calibration[0])


@pytest.mark.parametrize("output_format", ["nwb", "dh5"])
def test_export_cli_into_bids(work_dir, tmp_path, output_format):  # noqa: F811
    import json

    from ephysalis.cli.main import main

    root = tmp_path / "bids"
    assert main(["export", str(SPIKEGLX_CROP_SESSION), "--work-dir", str(work_dir), "--output-format",
                 output_format, "--bids", str(root)]) == 0
    ecephys = next(root.glob("sub-*/ses-*/ecephys"))
    data = next(ecephys.glob(f"*_ecephys.{output_format}"))
    prefix = data.name.removesuffix(f"_ecephys.{output_format}")
    for suffix in ("ecephys.json", "probes.tsv", "electrodes.tsv", "channels.tsv", "events.tsv"):
        assert (ecephys / f"{prefix}_{suffix}").exists(), suffix
    channels = pd.read_csv(ecephys / f"{prefix}_channels.tsv", sep="\t")
    assert len(channels) == 16 and set(channels.sampling_frequency) == {1000}
    if output_format == "dh5":
        assert channels.stream_id[0] == f"CONT{LFP_FIRST_CONT_ID}"
        assert (ecephys / f"{prefix}_ecephys_units.csv").exists()
        assert "*_ecephys_units.csv" in (root / ".bidsignore").read_text(encoding="utf-8")
    else:
        assert channels.stream_id[0].startswith("processing/ecephys/LFP/")
    assert json.loads((ecephys / f"{prefix}_ecephys.json").read_text(encoding="utf-8"))["SamplingFrequency"] == 1000
