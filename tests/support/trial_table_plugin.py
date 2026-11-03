"""Metadata plugin of the core tests: trials of a run from ``trial_table_<paradigm>.csv`` in the run
folder (trial times as samples of the auxiliary stream), so that the core tests of trials, events
and exports do not depend on a lab's plugin. Registered by ``tests/conftest.py``."""

from __future__ import annotations

import pandas as pd

from ephysalis.timeline.clocks import AUX

metadata_plugin_name = "trial_table"
plugin_api_version = 2


def extract_trial_tables(run, clock_alignment, auxiliary_signals):
    files = sorted(run.path.glob("trial_table_*.csv"))
    if not files:
        return None
    tables = {}
    for path in files:
        table = pd.read_csv(path)
        start, stop = table.pop("start_aux_sample").to_numpy(), table.pop("stop_aux_sample").to_numpy()
        if clock_alignment is not None:
            table.insert(1, "start_time", clock_alignment.to_reference_time(AUX, start))
            table.insert(2, "stop_time", clock_alignment.to_reference_time(AUX, stop))
        tables[path.stem.removeprefix("trial_table_")] = table
    return tables
