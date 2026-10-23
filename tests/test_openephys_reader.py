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
