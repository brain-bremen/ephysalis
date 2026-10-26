"""Run start on the stimulus PC's clock from the .tdr trial stamps (ephyscon.pcclock); BIDS task
entity and scans.tsv."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ephyscon.cli.main import main
from ephyscon.pcclock import PRIOR_PPM, pc_fit
from test_vstim_td01 import MS, T0, _session, write_tdr


def test_pc_fit():
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0, 1500, 120))  # trial starts on the run clock
    offset = 40_000.437
    c = np.floor(offset + (1 + PRIOR_PPM * 1e-6) * s)
    fit = pc_fit(c, s)
    assert fit["offset_s"] == pytest.approx(offset, abs=fit["window_ms"] / 2000 + 1e-9)
    assert fit["window_ms"] < 60 and not fit["tolerance_s"]
    c = np.floor(offset + (1 + 200e-6) * s)  # a rate the prior does not allow
    fit = pc_fit(c, s)
    assert not fit["rate_from_prior"] and fit["ppm_feasible"][0] <= 200 <= fit["ppm_feasible"][1]
    assert fit["offset_s"] == pytest.approx(offset, abs=0.03)
    few = pc_fit(c[:5], s[:5])  # poorly constrained: the prior rate
    assert few["rate_from_prior"] and few["ppm"] == PRIOR_PPM
    jitter = c + (rng.random(c.size) < 0.1) * 1.0  # some stamps 1 s late
    assert pc_fit(jitter, s)["tolerance_s"] > 0
    assert pc_fit(c[:2], s[:2]) is None


def test_run_start_task_and_scans(tmp_path):
    starts_ms = [1000 + 1530 * k for k in range(12)]
    pc_start = 11 * 3600 + 43 * 60 + 0.25  # the DAQ header (11:49:37) runs 396.75 s ahead
    stamps = [np.floor(pc_start + s / 1000 * (1 + 35e-6)) for s in starts_ms]
    events = []
    for s in starts_ms:
        events += [(T0 + s * MS, 1, 1), (T0 + (s + 300) * MS, -1, 0)]
    td01 = [(T0 + (s + 2) * MS, n, 1) for n, s in enumerate(starts_ms, start=1)]
    session = _session(tmp_path / "ses", sorted(events), td01)
    write_tdr(session / "2005_03_30.tdr", [(n, 1, 1, 1.0) for n in range(1, 13)],
              stamps=[f"{int(c) // 3600:02d}:{int(c) % 3600 // 60:02d}:{int(c) % 60:02d}" for c in stamps])
    (tmp_path / "session.toml").write_text(
        'schema_version = 5\nsession_id = "20050330"\nsubject_id = "fritz"\ntask = "shape tracking"\n'
        '[aux_channels.digital]\ntrial_start = 0\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["timing", str(session), *common]) == 0
    bids = tmp_path / "bids"
    assert main(["export", str(session), *common, "--per-run", "--output-format", "dh5", "--bids", str(bids)]) == 0
    data = next(bids.rglob("*_ecephys.dh5"))
    assert data.name == "sub-fritz_ses-20050330_task-shapetracking_run-01_ecephys.dh5"
    scans = pd.read_csv(next(bids.rglob("*_scans.tsv")), sep="\t")
    assert scans.filename.tolist() == [f"ecephys/{data.name}"]
    assert scans.start_time_method[0].startswith("PC clock from 12 .tdr trial stamps")
    acq_time = datetime.fromisoformat(scans.acq_time[0])
    expected = datetime(2005, 3, 30, 11, 43, 0, 250_000)
    assert abs((acq_time - expected).total_seconds()) < 0.1  # 12 trials: window of < 200 ms
    sessions = pd.read_csv(next(bids.rglob("*_sessions.tsv")), sep="\t")
    assert sessions.acq_time.tolist() == scans.acq_time.tolist()
    import h5py

    with h5py.File(data) as f:  # still on the DAQ clock
        assert f["Markers"]["trial_start"][0] == T0 + 1000 * MS


def test_frozen_td01_matched_by_time(tmp_path):
    """Run b's TD01 records all carry trial 99; run a of the same day gives the PC clock, and
    b's trials are matched to the day's .tdr by time."""
    from test_daq import write_v2

    header_ahead = 396.75  # DAQ computer clock minus PC clock
    session = tmp_path / "ses"
    session.mkdir()
    tdr_trials, stamps = [], []
    for k, (name, header) in enumerate((("task_a", (2005, 3, 30, 11, 49, 37)), ("task_b", (2005, 3, 30, 12, 10, 0)))):
        starts_ms = [1000 + 1530 * j for j in range(12)]
        pc_start = header[3] * 3600 + header[4] * 60 + header[5] - header_ahead
        events, td01 = [], []
        for j, s in enumerate(starts_ms):
            number = 1 + 12 * k + j
            events += [(T0 + s * MS, 1, 1), (T0 + (s + 300) * MS, -1, 0)]
            events += [(T0 + (s + 800) * MS, 14 if j % 3 else 11, 0), (T0 + (s + 900) * MS, -(14 if j % 3 else 11), 0)]
            td01.append((T0 + (s + 2) * MS, *((99, 7) if name == "task_b" else (number, 1 + j % 4))))
            tdr_trials.append((number, 1 + j % 4, 1 if j % 3 else 7, 1.0))
            stamps.append(np.floor(pc_start + s / 1000 * (1 + 35e-6)))
        write_v2(session / f"fritz_2005_03_30_{name}.daq", [(T0, np.zeros((20_000, 4), np.int16))],
                 events=sorted(events), trials=td01, date=header)
    write_tdr(session / "2005_03_30.tdr", tdr_trials,
              stamps=[f"{int(c) // 3600:02d}:{int(c) % 3600 // 60:02d}:{int(c) % 60:02d}" for c in stamps])
    (tmp_path / "session.toml").write_text(
        'schema_version = 5\nsession_id = "20050330"\nsubject_id = "fritz"\n'
        '[aux_channels.digital]\ntrial_start = 0\neye_error = 10\nreward = 13\n')
    common = ["--work-dir", str(tmp_path / "work"), "--config", str(tmp_path / "session.toml")]
    assert main(["timing", str(session), *common]) == 0
    trials = pd.read_csv(next((tmp_path / "work" / "trials").glob("fritz_2005_03_30_task_b_*.csv")))
    assert trials.vstim_trial.tolist() == list(range(13, 25))
    assert trials.condition.tolist() == [1 + j % 4 for j in range(12)] and trials.outcome_source.eq("tdr").all()
    assert trials.trial_number_source[0].startswith("TD01 frozen (all records trial 99): matched to the .tdr by time (12/12")
