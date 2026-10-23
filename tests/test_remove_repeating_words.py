from random import sample
import ephyscon
from ephyscon.events import (
    Event,
    EventMetadata,
    FullWordEvent,
    remove_repeating_simultaneous_words,
)
import numpy as np


def test_remove_repeating_words():
    metadata = EventMetadata(
        folder_name="test_folder",
        source_processor="PXIe-6341",
        stream_name="PXIe-6341",
        initial_state=0,
        identifier="",
        sample_rate=1000.0,
        channel_name="",
        type="",
        description="",
    )
    full_words = np.array([1, 3, 4, 4, 4, 128, 128, 129])
    states = np.array([1, 2, -1, -2, 3, -3, 8, 1])
    timestamps = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0])
    sample_numbers = np.array([1, 2, 3, 3, 3, 4, 4, 5])
    event_data = Event(
        states=states,
        full_words=full_words,
        timestamps=timestamps,
        metadata=metadata,
        sample_numbers=sample_numbers,
    )

    full_word_datat = remove_repeating_simultaneous_words(event_data)
    assert isinstance(full_word_datat, FullWordEvent)
    assert len(full_word_datat) == 5
    assert np.array_equal(full_word_datat.full_words, np.array([1, 3, 4, 128, 129]))
