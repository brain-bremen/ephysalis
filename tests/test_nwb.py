"""Session NWB file from the work-dir outputs of the 5 s SpikeGLX crop (timing, LFP and fake sorting outputs)."""

import numpy as np
import pandas as pd
import pytest
import spikeinterface.core as si
from pynwb import NWBHDF5IO

from conftest import SPIKEGLX_CROP_SESSION
from ephysalis.cli.main import main
from ephysalis.io.base import open_session
from ephysalis.nwb import SESSION_START_MARGIN_S, write_session_nwb
from ephysalis.session_config import ProbeConfig, SubjectConfig, default_session_config
from ephysalis.sorting import save_run_spikes
from ephysalis.sync import synchronize_run
from ephysalis.timing import aux_channel_map

RUN = "3_rfmappingFine_g0"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    assert main(["timing", str(SPIKEGLX_CROP_SESSION), "--work-dir", str(work)]) == 0
    assert main(["lfp", str(SPIKEGLX_CROP_SESSION), "--work-dir", str(work), "--n-jobs", "1"]) == 0
    # sorting outputs of imec0: units 3 (peak channel AP84) and 7 (AP89)
    (work / "sorting" / "imec0").mkdir(parents=True)
    pd.DataFrame({"unit_id": [3, 7], "KSLabel": ["good", "mua"], "peak_channel": ["imec0.ap#AP84", "imec0.ap#AP89"],
                  "y_um": [840.0, 880.0], "snr": [5.0, np.nan]}).to_csv(work / "sorting" / "imec0" / "units.csv", index=False)
    (work / "spikes").mkdir()
    save_run_spikes(work / "spikes" / f"{RUN}_imec0.npz",
                    pd.DataFrame({"time": [0.5, 0.1, 2.0], "sample": [15000, 3000, 60000], "unit_id": [3, 7, 3]}))
    return work


@pytest.fixture(scope="module")
def nwbfile(work_dir):
    session = open_session(SPIKEGLX_CROP_SESSION)
    config = default_session_config(session)
    config.subject = SubjectConfig(species="Mus musculus", sex="F", age="P90D")
    config.probes["imec0"] = ProbeConfig(area="V1", insertion_depth_um=850, bad_channels=[1])
    run = session.run(RUN)
    syncs = {RUN: synchronize_run(run, aux_channel_map(run, config))}
    path = write_session_nwb(session, work_dir, work_dir / "session.nwb", config, syncs)
    io = NWBHDF5IO(str(path), "r")
    yield io.read()
    io.close()


def test_time_axis_and_metadata(nwbfile):
    assert nwbfile.subject.species == "Mus musculus"
    assert nwbfile.session_start_time.second == 44  # run start 13:27:45 minus the margin
    epochs = nwbfile.epochs.to_dataframe()
    assert list(epochs.tags) == [[RUN]]
    assert epochs.start_time[0] == SESSION_START_MARGIN_S


def test_electrodes(nwbfile):
    electrodes = nwbfile.electrodes.to_dataframe()
    imec0 = electrodes[electrodes.group_name == "imec0"]
    assert len(electrodes) == 16
    assert list(imec0.channel_name[:3]) == ["imec0.ap#AP82", "imec0.ap#AP83", "imec0.ap#AP84"]
    assert list(imec0.in_brain) == [True] * 4 + [False] * 4  # y 820..880 um, depth 850 um
    assert list(imec0.bad_channel) == [False, True] + [False] * 6
    assert set(imec0.location) == {"V1"}


def test_trials_on_session_time_axis(nwbfile, work_dir):
    trials = nwbfile.trials.to_dataframe()
    csv = pd.read_csv(work_dir / "trials" / f"{RUN}_RF.csv")
    assert len(trials) == 23
    assert set(trials.paradigm) == {"RF"} and set(trials.run) == {RUN}
    np.testing.assert_allclose(trials.start_time, csv.start_time + SESSION_START_MARGIN_S)
    np.testing.assert_array_equal(trials.xc, csv.xc)


def test_units(nwbfile):
    units = nwbfile.units.to_dataframe()
    assert list(units.unit_id) == [3, 7] and list(units.probe) == ["imec0", "imec0"]
    np.testing.assert_allclose(units.spike_times[0], np.array([0.5, 2.0]) + SESSION_START_MARGIN_S)
    np.testing.assert_allclose(units.spike_times[1], [0.1 + SESSION_START_MARGIN_S])
    assert units.electrodes[0].channel_name.iloc[0] == "imec0.ap#AP84"
    assert units.KSLabel[0] == "good"


def test_lfp(nwbfile, work_dir):
    lfp = nwbfile.processing["ecephys"]["LFP"]
    for probe in ["imec0", "imec1"]:
        series = lfp[f"LFP_{RUN}_{probe}"]
        source = si.load(work_dir / "lfp" / f"{RUN}_{probe}")
        np.testing.assert_array_equal(series.data[:], source.get_traces(return_in_uV=False))
        uv = series.data[:100].astype(float) * series.conversion + series.offset
        np.testing.assert_allclose(uv * 1e6, source.get_traces(end_frame=100, return_in_uV=True), atol=1e-3)
        assert series.starting_time > 0
    # imec1 runs ~6.5 ppm fast relative to imec0 (sync pulses) and starts ~56 us earlier
    assert lfp[f"LFP_{RUN}_imec1"].rate == pytest.approx(1000.0065, abs=1e-4)
    assert lfp[f"LFP_{RUN}_imec0"].starting_time - lfp[f"LFP_{RUN}_imec1"].starting_time == pytest.approx(5.6e-5, abs=1e-5)


def test_running_speed(nwbfile, work_dir):
    series = nwbfile.processing["behavior"]["RunningSpeed"][f"running_speed_{RUN}"]
    np.testing.assert_array_equal(series.data[:], np.load(work_dir / "running_speed" / f"{RUN}.npy"))


def test_nwbinspector_finds_no_critical_issues(nwbfile, work_dir):
    nwbinspector = pytest.importorskip("nwbinspector")
    messages = list(nwbinspector.inspect_nwbfile(nwbfile_path=work_dir / "session.nwb"))
    critical = [m for m in messages if m.importance.name in ("CRITICAL", "PYNWB_VALIDATION")]
    assert not critical, critical
