"""The Open Ephys reader on all three recording formats (Binary, NWB, legacy Open Ephys) of the
open-ephys-python-tools test data, and Neuropixels probes from ``settings.xml``."""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import probeinterface
import pytest

from conftest import OE_TOOLS_DATA, skip_if_no_oe_tools_data
from ephyscon.cli.main import main
from ephyscon.io.base import open_session
from ephyscon.io.openephys import OpenEphysFormat
from ephyscon.session import StreamKind

FORMATS = ["v0.6.7_Binary", "v0.6.7_NWB", "v0.6.7_OpenEphys"]
NODE = "Record Node 105"
STREAM = f"{NODE}#File_Reader-100.example_data"
FIRST_SAMPLE = 40091  # Binary and NWB


def _run(fmt):
    return open_session(OE_TOOLS_DATA / fmt / NODE).runs[0]


@skip_if_no_oe_tools_data
@pytest.mark.parametrize("fmt", FORMATS)
def test_discovery(fmt):
    session = open_session(OE_TOOLS_DATA / fmt / NODE)
    assert isinstance(session.format, OpenEphysFormat)
    run = session.runs[0]
    assert [r.name for r in session.runs] == ["experiment1_recording1"]
    assert run.extra["streams"] == [STREAM] and run.aux_stream == STREAM and run.probes == {}
    recording = run.get_recording(StreamKind.AUX)
    assert recording.get_num_channels() == 16 and recording.sampling_frequency == 40000
    assert list(recording.channel_ids[:2]) == ["CH1", "CH2"]
    np.testing.assert_allclose(recording.get_channel_gains(), 0.05)
    assert run.duration_s == pytest.approx(recording.get_total_duration())
    assert run.start_time is not None


@skip_if_no_oe_tools_data
def test_binary_and_nwb_hold_the_same_samples():
    binary, nwb = (_run(fmt) for fmt in FORMATS[:2])
    a, b = (r.get_recording(StreamKind.AUX) for r in (binary, nwb))
    np.testing.assert_array_equal(a.get_traces(), b.get_traces())
    np.testing.assert_array_equal(a.get_traces(start_frame=1000, end_frame=1100, channel_ids=["CH3", "CH1"]),
                                  a.get_traces()[1000:1100, [2, 0]])
    ma, mb = (r.format.clock_maps(r).maps["aux"] for r in (binary, nwb))
    np.testing.assert_allclose(ma.reference_times, mb.reference_times)
    assert ma.to_reference_time(40000) == pytest.approx(1.0)


@skip_if_no_oe_tools_data
def test_legacy_samples_as_package_loader():
    """Lazy reads across the 1024-sample records equal open-ephys-python-tools' (in-memory) loader."""
    from open_ephys.analysis import Session

    run = _run(FORMATS[2])
    recording = run.get_recording(StreamKind.AUX)
    expected = Session(str(OE_TOOLS_DATA / FORMATS[2])).recordnodes[1].recordings[0].continuous[0].samples
    np.testing.assert_array_equal(recording.get_traces(), expected)
    np.testing.assert_array_equal(recording.get_traces(start_frame=1000, end_frame=3100, channel_ids=["CH16", "CH2"]),
                                  expected[1000:3100][:, [15, 1]])
    assert recording.get_traces(start_frame=5, end_frame=5).shape == (0, 16)


@skip_if_no_oe_tools_data
@pytest.mark.parametrize("fmt", FORMATS[:2])
def test_network_events_as_full_words(fmt):
    """Words rebuilt from the line states equal the full words the formats store."""
    run = _run(fmt)
    events = run.format.read_network_events(run)
    if fmt == "v0.6.7_Binary":
        folder = OE_TOOLS_DATA / fmt / NODE / "experiment1/recording1/events/Network_Events-108.example_data/TTL"
        words, samples = np.load(folder / "full_words.npy"), np.load(folder / "sample_numbers.npy")
    else:
        with h5py.File(OE_TOOLS_DATA / fmt / NODE / "experiment1.nwb") as f:
            group = f["acquisition/Network Events-108.example_data.TTL"]
            words, samples = group["full_word"][:, 0], group["sync"][()]
    last = np.r_[samples[1:] != samples[:-1], True]
    np.testing.assert_array_equal(events.word, words[last].astype(np.int64))
    np.testing.assert_array_equal(events.aux_sample, samples[last] - FIRST_SAMPLE)


@skip_if_no_oe_tools_data
@pytest.mark.parametrize("fmt", FORMATS)
def test_messages(fmt):
    run = _run(fmt)
    messages = run.format.read_messages(run)
    assert len(messages) >= 14 and messages.text.str.startswith(("TTL Line", "Start Time")).all()
    assert (messages.aux_sample >= 0).all() and messages.aux_sample.is_monotonic_increasing
    assert len(run.format.read_ttl_events(run)) == 0  # the File Reader stream has no TTL lines


@skip_if_no_oe_tools_data
@pytest.mark.parametrize("fmt", FORMATS)
def test_pipeline(fmt, tmp_path):
    """A probe defined in session.toml on the 16 channels, through LFP, MUA and both exports."""
    probe = probeinterface.generate_linear_probe(num_elec=16, ypitch=25)
    probeinterface.write_probeinterface(tmp_path / "probe.json", probe)
    (tmp_path / "session.toml").write_text(
        'schema_version = 3\nsession_id = "s"\nsubject_id = "m"\n'
        '[probes.probe0]\nstream = "File_Reader-100.example_data"\ngeometry = { file = "probe.json" }\n')
    session = str(OE_TOOLS_DATA / fmt / NODE)
    common = ["--work-dir", str(tmp_path), "--config", str(tmp_path / "session.toml")]
    assert main(["timing", session, *common]) == 0
    assert main(["lfp", session, *common, "--n-jobs", "1"]) == 0
    assert main(["mua", session, *common, "--n-jobs", "1"]) == 0
    assert main(["export", session, *common, "-o", str(tmp_path / "out.nwb")]) == 0
    assert main(["export", session, *common, "--output-format", "dh5", "--raw", "-o", str(tmp_path / "out.dh5")]) == 0
    events = pd.read_csv(tmp_path / "events" / "experiment1_recording1.csv")
    assert (events.source == "network").sum() == len(_run(fmt).format.read_network_events(_run(fmt)))
    with h5py.File(tmp_path / "out.dh5") as f:
        raw = f["CONT1"]["DATA"][:, 0]
    np.testing.assert_array_equal(raw, _run(fmt).get_recording(StreamKind.AUX).get_traces()[:, 0])


def _write_np_binary(root: Path, n_samples: int = 3000) -> Path:
    """Binary-format recording with a Neuropixels 1.0 probe (AP + LFP streams with a SYNC channel)
    and an NI-DAQ stream; settings.xml from the probeinterface test data (OE_Neuropix-PXI-NP1-binary)."""
    node = root / "Record Node 101"
    folder = node / "experiment1" / "recording1"
    (node).mkdir(parents=True)
    (node / "settings.xml").write_bytes((Path(__file__).parent / "data" / "openephys_np1_settings.xml").read_bytes())
    streams = [("Neuropix-PXI-100.ProbeA-AP", "ProbeA-AP", 100, 30000.0, [f"CH{i + 1}" for i in range(384)] + ["SYNC"], 0.195, "uV"),
               ("Neuropix-PXI-100.ProbeA-LFP", "ProbeA-LFP", 100, 2500.0, [f"CH{i + 1}" for i in range(384)] + ["SYNC"], 0.195, "uV"),
               ("NI-DAQmx-102.PXI-6221", "PXI-6221", 102, 30000.0, [f"AI{i}" for i in range(8)], 0.000305, "V")]
    continuous = []
    rng = np.random.default_rng(0)
    for name, stream, pid, rate, channels, bit_volts, units in streams:
        n = n_samples if rate == 30000.0 else n_samples // 12
        data_folder = folder / "continuous" / name
        data_folder.mkdir(parents=True)
        rng.integers(-500, 500, (n, len(channels)), dtype=np.int16).tofile(data_folder / "continuous.dat")
        np.save(data_folder / "sample_numbers.npy", np.arange(1000, 1000 + n, dtype=np.int64))
        np.save(data_folder / "timestamps.npy", 1000 / rate + np.arange(n) / rate)
        continuous.append({"folder_name": name + "/", "sample_rate": rate, "source_processor_name": name.rsplit("-", 1)[0],
                           "source_processor_id": pid, "stream_name": stream, "recorded_processor": name, "recorded_processor_id": pid,
                           "num_channels": len(channels),
                           "channels": [{"channel_name": c, "bit_volts": bit_volts, "units": units} for c in channels]})
    (folder / "structure.oebin").write_text(json.dumps({"GUI version": "0.6.7", "continuous": continuous, "events": [], "spikes": []}))
    (folder / "sync_messages.txt").write_text("Software Time (milliseconds since midnight Jan 1st 1970 UTC): 1773329528431\n")
    return root


def test_neuropixels_probe_from_settings(tmp_path):
    session = open_session(_write_np_binary(tmp_path / "np"))
    run = session.runs[0]
    assert session.probe_ids == ["ProbeA"] and run.probes["ProbeA"].has_lf
    assert run.aux_stream == "Record Node 101#NI-DAQmx-102.PXI-6221"
    ap = run.get_recording(StreamKind.AP, "ProbeA")
    assert ap.get_num_channels() == 384 and "SYNC" not in ap.channel_ids  # SYNC channel left out
    expected = probeinterface.read_openephys_neuropixels(tmp_path / "np" / "Record Node 101" / "settings.xml",
                                                         stream_name="Neuropix-PXI-100.ProbeA-AP")
    np.testing.assert_array_equal(ap.get_channel_locations(), expected.contact_positions[np.argsort(expected.device_channel_indices)])
    assert ap.get_property("inter_sample_shift") is not None
    raw = np.fromfile(tmp_path / "np" / "Record Node 101/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA-AP/continuous.dat",
                      dtype=np.int16).reshape(-1, 385)
    np.testing.assert_array_equal(ap.get_traces(start_frame=10, end_frame=20), raw[10:20, :384])
    lf = run.get_recording(StreamKind.LF, "ProbeA")
    assert lf.sampling_frequency == 2500 and lf.get_num_channels() == 384
    aux = run.get_recording(StreamKind.AUX)
    assert list(aux.channel_ids) == [f"AI{i}" for i in range(8)]
    assert run.format.aux_analog_gain_v(run, 2) == pytest.approx(0.000305)
    sync = run.format.clock_maps(run)
    assert set(sync.maps) == {"ProbeA", "aux"} and sync.maps["ProbeA"].to_reference_time(0) == 0.0


def test_recording_is_serializable(tmp_path):
    """Parallel steps (LFP, MUA) rebuild recordings in worker processes from their kwargs."""
    from spikeinterface.core import load

    run = open_session(_write_np_binary(tmp_path / "np")).runs[0]
    ap = run.get_recording(StreamKind.AP, "ProbeA")
    again = load(ap.to_dict())
    np.testing.assert_array_equal(again.get_traces(end_frame=50), ap.get_traces(end_frame=50))
    np.testing.assert_array_equal(again.get_channel_locations(), ap.get_channel_locations())
