from datetime import datetime

import pytest

from ephysalis.io.base import open_session
from ephysalis.io.spikeglx import SpikeGLXFormat, read_meta
from ephysalis.session import StreamKind
from conftest import SPIKEGLX_TEMPLATES


def test_read_meta_strips_tilde():
    meta = read_meta(SPIKEGLX_TEMPLATES / "template.imec0.ap.meta")
    assert meta["imDatPrb_pn"] == "PRB_1_4_0480_1_C"
    assert "snsGeomMap" in meta and "imroTbl" in meta


def test_runs_sorted_by_acquisition_time(make_spikeglx_session):
    session_path = make_spikeglx_session(
        runs=[
            ("10_OptoTag_g0", "2025-01-09T16:02:54"),
            ("3_rfmapping_g0", "2025-01-09T13:27:45"),
            ("5_DoseResponse_g0", "2025-01-09T13:58:53"),
        ]
    )
    session = open_session(session_path)
    assert isinstance(session.format, SpikeGLXFormat)
    assert [r.name for r in session.runs] == ["3_rfmapping_g0", "5_DoseResponse_g0", "10_OptoTag_g0"]
    assert [r.index for r in session.runs] == [0, 1, 2]
    assert session.runs[0].start_time == datetime(2025, 1, 9, 13, 27, 45)


def test_probe_and_aux_info(make_spikeglx_session):
    session = open_session(make_spikeglx_session())
    run = session.runs[0]
    assert session.probe_ids == ["imec0"]
    probe = run.probes["imec0"]
    assert probe.part_number == "PRB_1_4_0480_1_C"
    assert probe.num_shanks == 1
    assert probe.num_channels == 384
    assert probe.has_lf
    assert set(probe.streams) == {StreamKind.AP, StreamKind.LF, StreamKind.SYNC}
    assert run.aux_stream == "nidq"
    assert run.extra["aux"] == {"n_analog": 8, "n_digital_words": 1, "sync_bit": 1}


def test_get_recordings(make_spikeglx_session):
    run = open_session(make_spikeglx_session(duration_s=0.1)).runs[0]

    ap = run.get_recording(StreamKind.AP, "imec0")
    assert ap.get_num_channels() == 384
    assert ap.get_sampling_frequency() == pytest.approx(30000)
    assert ap.get_num_samples() == 3000
    assert ap.get_probe().get_contact_count() == 384

    lf = run.get_recording(StreamKind.LF, "imec0")
    assert lf.get_sampling_frequency() == pytest.approx(2500)

    sync = run.get_recording(StreamKind.SYNC, "imec0")
    assert sync.get_num_channels() == 1

    nidq = run.get_recording(StreamKind.AUX)
    assert nidq.get_num_channels() == 9


def test_missing_stream_raises(make_spikeglx_session):
    run = open_session(make_spikeglx_session(streams=("imec0.ap", "nidq"))).runs[0]
    assert not run.probes["imec0"].has_lf
    with pytest.raises(KeyError):
        run.get_recording(StreamKind.LF, "imec0")


def test_single_run_folder_is_a_session(make_spikeglx_session):
    session_path = make_spikeglx_session()
    session = open_session(session_path / "3_rfmapping_g0")
    assert [r.name for r in session.runs] == ["3_rfmapping_g0"]


def test_unknown_folder_is_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="Could not detect"):
        open_session(tmp_path / "empty")
