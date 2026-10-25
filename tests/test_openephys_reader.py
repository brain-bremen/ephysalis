import numpy as np
import pytest

from ephyscon.io.base import open_session
from ephyscon.io.openephys import OpenEphysFormat
from ephyscon.session import StreamKind
from conftest import DATA_FOLDER, skip_if_no_data


@skip_if_no_data
def test_open_ephys_session_discovery():
    session = open_session(DATA_FOLDER)
    assert isinstance(session.format, OpenEphysFormat)
    assert [r.name for r in session.runs] == ["experiment1_recording1"]
    run = session.runs[0]
    assert run.probes == {}
    assert run.aux_stream == "Record Node 101#NI-DAQmx-110.USB-6343"

    aux = run.get_recording(StreamKind.AUX)
    assert aux.get_num_channels() == 2
    assert aux.get_sampling_frequency() == 30000
    assert run.duration_s is not None and abs(run.duration_s - aux.get_total_duration()) < 0.01
    # sync_messages.txt: Software Time 1773329528431 ms (2026-03-12 15:32:08.431 UTC)
    from datetime import datetime, timezone

    assert run.start_time == datetime.fromtimestamp(1773329528.431)
    assert run.start_time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f") == "2026-03-12 15:32:08.431000"


@skip_if_no_data
def test_open_ephys_clock_maps_from_timestamps():
    from ephyscon.sync import AUX, synchronize_run

    run = open_session(DATA_FOLDER).runs[0]
    sync = synchronize_run(run)  # no probes and no sync pulses needed: Open Ephys timestamps
    assert sync.reference == AUX
    clock = sync.maps[AUX]
    assert clock.to_reference_time(0) == 0.0
    assert clock.to_reference_time(30000) == pytest.approx(1.0)
    assert clock.effective_sampling_frequency == pytest.approx(30000, rel=1e-6)


@skip_if_no_data
def test_probe_defined_in_session_config(tmp_path):
    """A non-Neuropixels probe on a stream the reader does not detect (here the two NI-DAQ inputs)."""
    import probeinterface

    from ephyscon.io.probes import configure_probes
    from ephyscon.session_config import ProbeConfig, ProbeGeometry, SessionConfig
    from ephyscon.sync import synchronize_run

    probe = probeinterface.generate_linear_probe(num_elec=2, ypitch=50)
    probeinterface.write_probeinterface(tmp_path / "probe.json", probe)
    config = SessionConfig(session_id="s", subject_id="m")
    config.probes["probe0"] = ProbeConfig(stream="NI-DAQmx-110.USB-6343", channels=["AI2", "AI0"],
                                          geometry=ProbeGeometry(file="probe.json"))
    session = configure_probes(open_session(DATA_FOLDER), config, tmp_path)
    assert session.probe_ids == ["probe0"]
    run = session.runs[0]
    recording = run.get_recording(StreamKind.AP, "probe0")
    assert list(recording.channel_ids) == ["AI2", "AI0"]
    np.testing.assert_array_equal(recording.get_channel_locations()[:, 1], [0, 50])
    sync = synchronize_run(run)
    assert sync.reference == "probe0" and sync.maps["probe0"].to_reference_time(0) == 0.0
