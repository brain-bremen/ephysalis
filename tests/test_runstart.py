"""Better run start times (ephysalis.runstart), BIDS task entity and scans.tsv."""

from datetime import timedelta

import pandas as pd

from ephysalis.cli.main import main
from ephysalis.io.base import open_session
from ephysalis.runstart import RunStart, apply_run_starts, write_run_start
from synthetic import write_openephys


def test_run_starts_task_and_scans(tmp_path):
    session_path = write_openephys(tmp_path / "ses", [{"data": [[0]] * 3000}, {"data": [[0]] * 2000}])
    work = tmp_path / "work"
    session = open_session(session_path)
    first, second = session.runs
    recorded = [first.start_time, second.start_time]
    better = recorded[0] - timedelta(seconds=396.75)  # e.g. on the stimulus PC's clock
    write_run_start(work, first.name, first.start_time, RunStart(better, "from the stimulus clock", {"n": 1}))
    apply_run_starts(session, work)
    assert first.start_time == better and first.recorded_start_time == recorded[0]
    assert first.start_time_method == "from the stimulus clock"
    assert second.start_time == recorded[1] - timedelta(seconds=396.75)  # the day's offset of the other run
    assert "day's clock offset 396.8 s" in second.start_time_method
    apply_run_starts(session, work)  # idempotent: keeps the recorded times
    assert first.recorded_start_time == recorded[0] and second.start_time == recorded[1] - timedelta(seconds=396.75)

    (tmp_path / "session.toml").write_text(
        'schema_version = 5\nsession_id = "20231114"\nsubject_id = "m"\ntask = "shape tracking"\n'
        f'[[runs]]\nname = "{second.name}"\ntask = "rf mapping"\n')
    common = ["--work-dir", str(work), "--config", str(tmp_path / "session.toml")]
    bids = tmp_path / "bids"
    assert main(["export", str(session_path), *common, "--per-run", "--output-format", "dh5", "--bids", str(bids)]) == 0
    names = sorted(p.name for p in bids.rglob("*_ecephys.dh5"))
    assert names == ["sub-m_ses-20231114_task-rfmapping_run-02_ecephys.dh5",
                     "sub-m_ses-20231114_task-shapetracking_run-01_ecephys.dh5"]
    scans = pd.read_csv(next(bids.rglob("*_scans.tsv")), sep="\t").set_index("runs")
    assert scans.acq_time[first.name] == better.isoformat(timespec="milliseconds")
    assert scans.start_time_method[first.name] == "from the stimulus clock"
    sessions = pd.read_csv(next(bids.rglob("*_sessions.tsv")), sep="\t")
    assert sessions.acq_time.tolist() == [better.isoformat(timespec="milliseconds")]  # the earliest file
