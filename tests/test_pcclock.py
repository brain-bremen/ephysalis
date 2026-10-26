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
