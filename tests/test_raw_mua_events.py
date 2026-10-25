"""Raw data, MUA and events in DH5/NWB, against the DH5 file of the old ``ephyscon convert``
(Open Ephys test session) and on the 5 s SpikeGLX crop; several sessions per command."""

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.signal as signal
import spikeinterface.core as si

import dh5io
from conftest import DATA_FOLDER, LEGACY_DH5_PATH, SPIKEGLX_CROP_SESSION, skip_if_no_data
from ephyscon.cli.main import main
from ephyscon.dh5 import ANALOG_FIRST_CONT_ID, MUA_FIRST_CONT_ID, RAW_FIRST_CONT_ID
from ephyscon.events import load_run_events
from ephyscon.io.base import open_session
from ephyscon.mua import mua_folder

OE_RUN = "experiment1_recording1"
CROP_RUN = "3_rfmappingFine_g0"
needs_dh5_spikes = pytest.mark.skipif(not hasattr(dh5io, "Spike"), reason="needs dh5io with SPIKE block writing")


@pytest.fixture(scope="module")
def oe_dh5(tmp_path_factory):
    work = tmp_path_factory.mktemp("oe")
    assert main(["timing", str(DATA_FOLDER), "--work-dir", str(work)]) == 0
    path = work / "oe.dh5"
    assert main(["export", str(DATA_FOLDER), "--work-dir", str(work), "--output-format", "dh5", "--raw",
                 "-o", str(path)]) == 0
    return work, path


@skip_if_no_data
@needs_dh5_spikes
def test_events_as_legacy_ev02(oe_dh5):
    work, path = oe_dh5
    events = load_run_events(work / "events" / f"{OE_RUN}.csv")
    assert set(events.source) == {"ttl", "network"}
    with h5py.File(path, "r") as new, h5py.File(LEGACY_DH5_PATH, "r") as legacy:
        ev, old = new["EV02"][:], legacy["EV02"][:]
        network = ev[ev["event"] >= 1000]  # the old converter found no TTL lines (it looked for a PXIe-6341)
        np.testing.assert_array_equal(network["event"], old["event"])
        shift = network["time"] - old["time"]  # other time axis: session start instead of acquisition start
        assert np.ptp(shift) <= 1
        ttl = ev[ev["event"] < 1000]
        assert set(np.unique(ttl["event"])) == {1, -1, 2, -2, 3, -3, 4, -4, 8, -8}
        assert len(ttl) == (events.source == "ttl").sum()
        attrs = new["EV02"].attrs
        assert attrs["TrialStarted"] == old["event"][0] + 10 and attrs["VtlOn"] == 2000
        assert attrs["trial_startOn"] == 1 and attrs["frameOff"] == -8


@skip_if_no_data
@needs_dh5_spikes
def test_aux_raw_as_legacy_raw(oe_dh5):
    _, path = oe_dh5
    with h5py.File(path, "r") as new, h5py.File(LEGACY_DH5_PATH, "r") as legacy:
        for k in range(2):  # AI0, AI2: legacy CONT1/2, now in the analog range
            mine, old = new[f"CONT{ANALOG_FIRST_CONT_ID + k}"], legacy[f"CONT{RAW_FIRST_CONT_ID + k}"]
            np.testing.assert_array_equal(mine["DATA"][:], old["DATA"][:])
            assert mine.attrs["Calibration"] == pytest.approx(old.attrs["Calibration"])
            assert mine.attrs["SamplePeriod"] == old.attrs["SamplePeriod"] == 33333
            # the old converter wrote one region; 33333 ns is not 1/30 kHz, so there are more now
            regions = mine["INDEX"][:]
            assert len(regions) > 1
            true_time = regions["time"][0] + regions["offset"] / 30000 * 1e9
            np.testing.assert_allclose(regions["time"], true_time, atol=1)


@pytest.fixture(scope="module")
def crop_work(tmp_path_factory):
    work = tmp_path_factory.mktemp("crop")
    assert main(["timing", str(SPIKEGLX_CROP_SESSION), "--work-dir", str(work)]) == 0
    assert main(["mua", str(SPIKEGLX_CROP_SESSION), "--work-dir", str(work), "--n-jobs", "1",
                 "--probes", "imec0"]) == 0
    return work


def test_mua_matches_legacy_algorithm(crop_work):
    """filtfilt high-pass, rectification and anti-aliased decimation, as in the old converter."""
    mua = si.load(mua_folder(crop_work, CROP_RUN, "imec0"))
    ap = open_session(SPIKEGLX_CROP_SESSION).run(CROP_RUN).get_recording("ap", "imec0")
    assert mua.sampling_frequency == 1000 and mua.get_num_channels() == ap.get_num_channels()
    channel = 5
    x = ap.get_traces(channel_ids=[ap.channel_ids[channel]], return_in_uV=True)[:, 0].astype(float)
    b, a = signal.butter(4, 300, "highpass", fs=ap.sampling_frequency)
    legacy = signal.decimate(np.abs(signal.filtfilt(b, a, x)), 30, n=600, ftype="fir", zero_phase=True)
    mine = mua.get_traces()[:, channel]
    n = min(len(mine), len(legacy))
    middle = slice(200, n - 200)
    assert np.corrcoef(mine[middle], legacy[middle])[0, 1] > 0.99
    assert np.median(mine[middle] / legacy[middle]) == pytest.approx(1, rel=0.05)


def test_mua_and_events_in_nwb(crop_work):
    from pynwb import NWBHDF5IO

    path = crop_work / "crop.nwb"
    assert main(["export", str(SPIKEGLX_CROP_SESSION), "--work-dir", str(crop_work), "-o", str(path), "--raw"]) == 0
    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        series = nwbfile.processing["ecephys"]["MUA"][f"MUA_{CROP_RUN}_imec0"]
        n_channels = open_session(SPIKEGLX_CROP_SESSION).run(CROP_RUN).get_recording("ap", "imec0").get_num_channels()
        assert series.rate == pytest.approx(1000, rel=1e-4) and series.data.shape[1] == n_channels
        events = nwbfile.processing["events"]["events"].to_dataframe()
        assert (events.line == 7).all() and len(events) == 22
        raw = nwbfile.acquisition[f"ElectricalSeries_{CROP_RUN}_imec0"]
        assert raw.data.shape == (open_session(SPIKEGLX_CROP_SESSION).run(CROP_RUN).get_recording("ap", "imec0")
                                  .get_num_samples(), n_channels)
        aux = nwbfile.acquisition[f"aux_analog_{CROP_RUN}"]
        assert aux.unit == "volts" and aux.data.shape[1] == 8


@needs_dh5_spikes
def test_mua_in_dh5(crop_work):
    path = crop_work / "crop.dh5"
    assert main(["export", str(SPIKEGLX_CROP_SESSION), "--work-dir", str(crop_work), "--output-format", "dh5",
                 "-o", str(path)]) == 0
    mua = si.load(mua_folder(crop_work, CROP_RUN, "imec0"))
    with h5py.File(path, "r") as file:
        block = file[f"CONT{MUA_FIRST_CONT_ID + 5}"]
        assert block.attrs["SignalType"] == b"MUA/ESA" or block.attrs["SignalType"] == "MUA/ESA"
        volts = block["DATA"][:, 0] * block.attrs["Calibration"]
        np.testing.assert_allclose(volts, mua.get_traces()[:, 5] * 1e-6, atol=float(block.attrs["Calibration"]))
        assert f"CONT{MUA_FIRST_CONT_ID + mua.get_num_channels()}" not in file  # imec1 has no MUA
        assert "EV02" in file


@skip_if_no_data
def test_several_sessions(tmp_path):
    sessions = [str(SPIKEGLX_CROP_SESSION), str(DATA_FOLDER)]
    assert main(["timing", *sessions, "--work-dir", str(tmp_path / "w")]) == 2  # needs {session}
    assert main(["timing", *sessions, "--work-dir", str(tmp_path / "{session}")]) == 0
    for session in (SPIKEGLX_CROP_SESSION, DATA_FOLDER):
        assert (tmp_path / session.name / "trials").is_dir()
        assert len(pd.read_csv(next((tmp_path / session.name / "events").glob("*.csv"))))
    # a failing session does not stop the others
    assert main(["timing", str(tmp_path / "missing"), str(SPIKEGLX_CROP_SESSION),
                 "--work-dir", str(tmp_path / "again_{session}")]) == 1
    assert (tmp_path / f"again_{SPIKEGLX_CROP_SESSION.name}" / "trials").is_dir()
