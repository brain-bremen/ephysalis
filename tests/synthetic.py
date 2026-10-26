"""Synthetic Open Ephys Binary sessions for the tests of format-independent steps (calibration,
mains removal, CSD, channel selection, annotations): one NI-DAQ stream with analog inputs
``AI<n>`` and TTL lines, one recording (= run) per item of ``recordings``."""

import json
from pathlib import Path

import numpy as np

STREAM = "NI-DAQmx-110.USB-6343"
BIT_VOLTS = 1e-6  # 1 µV per count (gains in µV per count = 1)


def write_openephys(root: Path, recordings: list[dict], rate: float = 1000.0) -> Path:
    """``recordings``: dicts with ``data`` (samples x channels, int16 counts), optional ``ttl``
    [(sample, line, state)] and ``start_ms`` (ms since 1970, default 1.7e12 + 100 s per recording)."""
    node = root / "Record Node 101"
    for k, recording in enumerate(recordings):
        folder = node / "experiment1" / f"recording{k + 1}"
        data = np.asarray(recording["data"], dtype=np.int16)
        n, n_channels = data.shape
        first = 1000 * (k + 1)
        continuous = folder / "continuous" / STREAM
        continuous.mkdir(parents=True)
        data.tofile(continuous / "continuous.dat")
        np.save(continuous / "sample_numbers.npy", np.arange(first, first + n, dtype=np.int64))
        np.save(continuous / "timestamps.npy", (first + np.arange(n)) / rate)
        ttl = sorted(recording.get("ttl", []))
        events = folder / "events" / STREAM / "TTL"
        events.mkdir(parents=True)
        samples = np.array([s for s, _, _ in ttl], dtype=np.int64) + first
        np.save(events / "sample_numbers.npy", samples)
        np.save(events / "timestamps.npy", samples / rate)
        np.save(events / "states.npy", np.array([line if state else -line for _, line, state in ttl], dtype=np.int16))
        words, word = [], 0
        for _, line, state in ttl:
            word = word | (1 << (line - 1)) if state else word & ~(1 << (line - 1))
            words.append(word)
        np.save(events / "full_words.npy", np.array(words, dtype=np.uint64))
        oebin = {
            "GUI version": "0.6.7",
            "continuous": [{"folder_name": f"{STREAM}/", "sample_rate": rate, "source_processor_name": "NI-DAQmx",
                            "source_processor_id": 110, "stream_name": "USB-6343", "recorded_processor": "Record Node",
                            "recorded_processor_id": 101, "num_channels": n_channels,
                            "channels": [{"channel_name": f"AI{c}", "bit_volts": BIT_VOLTS, "units": "V"}
                                         for c in range(n_channels)]}],
            "events": [{"folder_name": f"{STREAM}/TTL/", "channel_name": "USB-6343Digital Input Line",
                        "description": "Digital Line", "identifier": "identifier", "sample_rate": rate, "type": "int16",
                        "source_processor": "NI-DAQmx", "stream_name": "USB-6343", "initial_state": 0}],
            "spikes": [],
        }
        (folder / "structure.oebin").write_text(json.dumps(oebin))
        start_ms = recording.get("start_ms", 1_700_000_000_000 + 100_000 * k)
        (folder / "sync_messages.txt").write_text(
            f"Software Time (milliseconds since midnight Jan 1st 1970 UTC): {start_ms}\n"
            f"Start Time for NI-DAQmx (110) - USB-6343 @ {rate:g} Hz: {first}\n")
    return root
