from enum import StrEnum


class ContGroups(StrEnum):
    RAW = "RAW"
    ANALOG = "ANALOG"
    LFP = "LFP"
    ESA = "ESA"
    AP = "AP"


DEFAULT_CONT_GROUP_RANGES = {
    ContGroups.RAW: (
        1,
        1600,
    ),  # room for 4 x 384 = 1536 channels from Neuropixel probe
    ContGroups.ANALOG: (1601, 2000),
    # downsampled signals
    ContGroups.LFP: (2001, 4000),
    ContGroups.ESA: (4001, 6000),
    # high-pass filtered signals (not downsamples, should not be used for long-term storage)
    ContGroups.AP: (6001, 8000),
}
