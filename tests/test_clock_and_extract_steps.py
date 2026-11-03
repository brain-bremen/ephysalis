"""align-clocks and extract-trials / -events / -behavior: each writes only its own outputs."""

import json

import numpy as np
import pandas as pd

from conftest import SPIKEGLX_CROP_SESSION
from ephysalis.cli.main import main
from ephysalis.events import COLUMNS, dh5_event_codes, load_run_events
from ephysalis.provenance import CLOCK_AND_EXTRACT_STEPS

RUN = "3_rfmappingFine_g0"


def test_steps_write_their_own_outputs(tmp_path):
    session = str(SPIKEGLX_CROP_SESSION)
    common = ["--work-dir", str(tmp_path)]
    assert main(["align-clocks", session, *common]) == 0
    clocks = json.loads((tmp_path / "clocks" / f"{RUN}.json").read_text())
    assert clocks["reference"] in clocks["streams"] and len(clocks["streams"]) >= 2
    assert all(s["sync_edges"] >= 2 and s["max_residual_ms"] < 1 for s in clocks["streams"].values())
    assert not (tmp_path / "trials").exists() and not (tmp_path / "events").exists()

    assert main(["extract-events", session, *common]) == 0
    assert (tmp_path / "events").is_dir() and not (tmp_path / "trials").exists()
    assert list(pd.read_csv(tmp_path / "events" / f"{RUN}.csv").columns) == COLUMNS  # no DH5 codes

    assert main(["extract-trials", session, *common]) == 0
    assert len(pd.read_csv(tmp_path / "trials" / f"{RUN}_RF.csv")) == 23
    assert main(["extract-behavior", session, *common]) == 0
    assert {p.stem for p in (tmp_path / "provenance").glob("*.json")} == set(CLOCK_AND_EXTRACT_STEPS)


def test_dh5_event_codes(tmp_path):
    events = pd.DataFrame({"time": [0.1, 0.2, 0.3], "source": ["ttl", "ttl", "network"],
                           "line": [3, 3, None], "state": [1, 0, None], "word": [None, None, 7],
                           "name": ["a", "a", "x"]})
    events.to_csv(tmp_path / "events.csv", index=False)
    loaded = load_run_events(tmp_path / "events.csv")
    assert list(loaded.columns) == COLUMNS
    np.testing.assert_array_equal(dh5_event_codes(loaded, 1000), [3, -3, 1007])
