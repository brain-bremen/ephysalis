import numpy as np
import pytest
from pathlib import Path
from dh5io import DH5File

from conftest import skip_if_no_data, DATA_FOLDER, GOLDEN_DH5_PATH
from ephyscon.events import EventMetadata, Event, Messages, FullWordEvent, event_from_eventfolder


@skip_if_no_data
def test_load_events(oe_recording):
    events = oe_recording.info["events"]
    for event in events:
        metadata = EventMetadata(**event)
        event_data = event_from_eventfolder(
            recording_directory=oe_recording.directory,
            metadata=metadata,
        )
        assert isinstance(event_data, (Event, Messages, FullWordEvent))


@skip_if_no_data
def test_regression_matches_golden(fresh_dh5_path, golden_dh5):
    """Conversion with the same config must reproduce the golden DH5 exactly."""
    fresh = DH5File(str(fresh_dh5_path))
    try:
        assert set(fresh.get_cont_group_ids()) == set(golden_dh5.get_cont_group_ids())

        for cont_id in golden_dh5.get_cont_group_ids():
            golden_data = golden_dh5.get_cont_data_by_id(cont_id)
            fresh_data = fresh.get_cont_data_by_id(cont_id)
            assert np.array_equal(golden_data, fresh_data), f"CONT {cont_id} data mismatch"

        assert np.array_equal(golden_dh5.get_events_array(), fresh.get_events_array())
        assert np.array_equal(golden_dh5.get_trialmap().recarray, fresh.get_trialmap().recarray)
    finally:
        fresh._file.close()
