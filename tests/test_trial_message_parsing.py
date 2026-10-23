import pytest
from ephyscon.trialmap import (
    TrialStartMessage,
    TrialEndMessage,
    parse_trial_start_message,
    parse_trial_end_message,
    parse_message,
)
from vstim.tdr import TrialOutcome


def test_parse_trial_start_message():
    # without prefix
    message = "TRIAL_START 1 TRIALTYPE 0 TIMESEQUENCE 0 FRAME 1032"
    data = parse_trial_start_message(message)
    assert data.frame_number == 1032
    assert data.trial_index == 1
    assert data.trial_type_number == 0
    assert data.time_sequence_index == 0

    # different order
    message = "TRIALTYPE 0 TIMESEQUENCE 0 FRAME 1032 TRIAL_START 1"
    data = parse_trial_start_message(message)
    assert data.frame_number == 1032
    assert data.trial_index == 1
    assert data.trial_type_number == 0
    assert data.time_sequence_index == 0

    # broken message
    with pytest.raises(ValueError):
        parse_trial_start_message(message[3:20])

    # warn about unknown key
    message = "TRIAL_START 1 TRIALTYPE 0 TIMESEQUENCE 0 FRAME 1032 FOO 1234"
    with pytest.warns(UserWarning):
        parse_trial_start_message(message)


def test_parse_trial_end_message():
    # without prefix
    message = "TRIAL_END 1 TRIALTYPE 0 OUTCOME 1 FRAME 2048"
    data = parse_trial_end_message(message)
    assert data.frame_number == 2048
    assert data.trial_index == 1
    assert data.trial_type_number == 0
    assert data.outcome == TrialOutcome.Hit

    # different order
    message = "TRIALTYPE 0 OUTCOME 1 FRAME 2048 TRIAL_END 1"
    data = parse_trial_end_message(message)
    assert data.frame_number == 2048
    assert data.trial_index == 1
    assert data.trial_type_number == 0
    assert data.outcome == TrialOutcome.Hit

    # broken message
    with pytest.raises(ValueError):
        parse_trial_end_message(message[3:20])

    # warn about unknown key
    message = "TRIAL_END 1 TRIALTYPE 0 OUTCOME 1 FRAME 2048 FOO 5678"
    with pytest.warns(UserWarning):
        parse_trial_end_message(message)


def test_message_parsing():
    # test selecting the right parsing function
    message = "VSTIM: TRIAL_START 1 TRIALTYPE 0 TIMESEQUENCE 0 FRAME 1032"
    assert isinstance(parse_message(message), TrialStartMessage)

    message = "VSTIM: TRIAL_END 1 TRIALTYPE 0 OUTCOME 0 FRAME 1032"
    assert isinstance(parse_message(message), TrialEndMessage)

    message = "VSTIM: TRIAL_ASSDLKJASDEND 1 TRIALTYPE 0 TIMESEQUENCE 0 FRAME 1032"
    assert parse_message(message) is None
