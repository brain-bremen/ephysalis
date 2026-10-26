"""Pulses and markers of the trigger lines (ephyscon.annotations) and their DH5 / NWB / BIDS output."""

import h5py
import numpy as np
import pandas as pd
import pytest

from ephyscon.annotations import event_code_names, line_pulses, markers
from ephyscon.cli.main import main
from ephyscon.session_config import EventsConfig, MarkerConfig, TriggerLineConfig
from test_vstim_td01 import MS, T0, _session, write_tdr


def _events(rows):
    return pd.DataFrame([{"time": t, "source": "ttl", "line": line, "state": state, "name": name}
                         for t, line, state, name in rows])


def test_line_pulses():
    config = EventsConfig(lines={"lever_pressed": TriggerLineConfig(active="low")})
    events = _events([(0.5, 15, 1, "lever_pressed"),  # release before the first press: dropped
                      (1.0, 1, 1, "interval0"), (1.5, 1, 0, "interval0"),
                      (2.0, 15, 0, "lever_pressed"), (2.7, 15, 1, "lever_pressed"),
                      (3.0, 1, 1, "interval0"), (3.2, 1, 1, "interval0"),  # two starts: the first has no end
                      (4.0, 15, 0, "lever_pressed")])  # no release before the end of the run
    pulses = line_pulses(events, config, "a")
    assert pulses.role.tolist() == ["interval0", "lever_pressed", "interval0", "interval0", "lever_pressed"]
    np.testing.assert_allclose(pulses.start_time, [1.0, 2.0, 3.0, 3.2, 4.0])
    np.testing.assert_allclose(pulses.stop_time, [1.5, 2.7, np.nan, np.nan, np.nan])
    assert set(pulses.run) == {"a"}
    assert event_code_names({1: "interval0", 15: "lever_pressed"}, config) == [
        (-15, "lever_pressed_on"), (-1, "interval0_off"), (1, "interval0_on"), (15, "lever_pressed_off")]


def test_markers():
    config = EventsConfig(markers=[
        MarkerConfig(name="lever_release", role="lever_pressed", edge="offset"),
        MarkerConfig(name="derived_sample_onset", role="interval2", offset_ms=650, description="sample on"),
        MarkerConfig(name="trial_end", role="trial", edge="offset")])
    pulses = pd.DataFrame({"start_time": [1.0, 2.0, 5.0], "stop_time": [1.5, 2.5, np.nan],
                           "role": ["lever_pressed", "interval2", "lever_pressed"], "line": [15, 3, 15]})
    trials = pd.DataFrame({"start_time": [0.5, 4.0], "stop_time": [3.0, 6.0]})
    found = markers(pulses, trials, config)
    assert list(found) == ["trial_start", "lever_release", "derived_sample_onset", "trial_end"]
    np.testing.assert_allclose(found["lever_release"][0], [1.5])  # the open pulse has no end
    np.testing.assert_allclose(found["derived_sample_onset"][0], [2.65])
    assert "derived" in found["derived_sample_onset"][1] and found["derived_sample_onset"][1].startswith("sample on")
    np.testing.assert_allclose(found["trial_start"][0], [0.5, 4.0])
    np.testing.assert_allclose(found["trial_end"][0], [3.0, 6.0])


def test_dh5_nwb_bids_annotations(tmp_path):
    starts = [1000, 6000]
    events = []
    for s in starts:
        events += [(T0 + s * MS, 1, 1), (T0 + (s + 300) * MS, -1, 0), (T0 + (s + 400) * MS, 3, 1),
                   (T0 + (s + 2000) * MS, -3, 0), (T0 + (s + 500) * MS, -15, 0), (T0 + (s + 2500) * MS, 15, 0)]
    td01 = [(T0 + (s + 2) * MS, n, 1) for s, n in zip(starts, (1, 2))]
    session = _session(tmp_path / "ses", sorted(events), td01)
    write_tdr(session / "2005_03_30.tdr", [(1, 1, 1, 3.0), (2, 1, 7, 3.0)])
    (tmp_path / "session.toml").write_text(
        'schema_version = 5\nsession_id = "s"\nsubject_id = "fritz"\n'
        '[aux_channels.digital]\ntrial_start = 0\ninterval2 = 2\nlever_pressed = 14\n'
        '[events.lines.lever_pressed]\nactive = "low"\ndescription = "lever pressed"\n'
        '[[events.markers]]\nname = "lever_release"\nrole = "lever_pressed"\nedge = "offset"\n'
        '[[events.markers]]\nname = "derived_sample_onset"\nrole = "interval2"\noffset_ms = 650\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["timing", str(session), *common]) == 0
    assert main(["export", str(session), *common, "--output-format", "dh5", "--per-run", "-o",
                 str(tmp_path / "out.dh5")]) == 0
    t0 = T0 / 1e9  # DAQ clock
    with h5py.File(next(tmp_path.glob("out_*.dh5"))) as f:
        lever = f["Intervals"]["lever_pressed"][()]
        np.testing.assert_array_equal(lever["start_time"], [T0 + 1500 * MS, T0 + 6500 * MS])
        np.testing.assert_array_equal(lever["end_time"], [T0 + 3500 * MS, T0 + 8500 * MS])
        assert "active low" in f["Intervals"]["lever_pressed"].attrs["Source"]
        assert f["Intervals"]["run_fritz_2005_03_30_task_a"][0]["start_time"] == T0
        np.testing.assert_array_equal(f["Markers"]["trial_start"][()], [T0 + 1000 * MS, T0 + 6000 * MS])
        np.testing.assert_array_equal(f["Markers"]["lever_release"][()], [T0 + 3500 * MS, T0 + 8500 * MS])
        np.testing.assert_array_equal(f["Markers"]["derived_sample_onset"][()], [T0 + 2050 * MS, T0 + 7050 * MS])
        legend = dict(zip(f["EV02"].attrs["EventNames"].astype(str), f["EV02"].attrs["EventCodes"]))
        assert legend["lever_pressed_on"] == -15 and legend["trial_start_on"] == 1
    bids = tmp_path / "bids"
    assert main(["export", str(session), *common, "--per-run", "--bids", str(bids)]) == 0
    from pynwb import NWBHDF5IO

    nwb_path = next(bids.rglob("*_ecephys.nwb"))
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()
        pulses = nwbfile.intervals["trigger_pulses"].to_dataframe()
        assert pulses[pulses.role == "interval2"].start_time.tolist() == pytest.approx([t0 + 1.4, t0 + 6.4])
        names = nwbfile.processing["events"]["markers"].to_dataframe().label
        assert set(names) == {"trial_start", "lever_release", "derived_sample_onset"}
    table = pd.read_csv(next(bids.rglob("*_events.tsv")), sep="\t")
    lever_rows = table[table.trial_type == "lever_pressed"]
    assert lever_rows.duration.tolist() == pytest.approx([2.0, 2.0]) and set(lever_rows.ttl_line) == {15}
    import json

    levels = json.loads(next(bids.rglob("*_events.json")).read_text())["trial_type"]["Levels"]
    assert levels["lever_pressed"].startswith("lever pressed (TTL line 15, active low")
