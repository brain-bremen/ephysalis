"""Session pipeline page of the GUI (offscreen), on the 5 s SpikeGLX crop."""

import os
import sys
import time

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import SPIKEGLX_CROP_SESSION  # noqa: E402
from ephyscon.gui import session_page  # noqa: E402
from ephyscon.gui.pipeline import PipelineOptions, step_commands, work_dir_summary  # noqa: E402
from ephyscon.session_config import load_session_config, save_session_config  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(app, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(session_page, "_settings", lambda: settings)
    widget = session_page.SessionPipelinePage()
    widget._session_edit.setText(str(SPIKEGLX_CROP_SESSION))
    widget._work_edit.setText(str(tmp_path / "work"))
    yield widget
    widget.deleteLater()


def _wait(app, condition, timeout_s=120.0):
    end = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > end:
            raise TimeoutError("condition not reached")
        app.processEvents()
        time.sleep(0.02)


def test_step_commands(tmp_path):
    options = PipelineOptions(session=tmp_path / "s", work_dir=tmp_path / "w", config=tmp_path / "w" / "session.toml",
                              steps=["export", "sort", "sync"], sorter="mountainsort5", output_format="dh5",
                              bids_root=tmp_path / "bids", export_lfp=False, scratch_dir=tmp_path / "scratch")
    commands = step_commands(options)
    assert [step for step, _ in commands] == ["sync", "sort", "export"]  # pipeline order
    assert all(argv[:4] == [sys.executable, "-u", "-m", "ephyscon.cli.main"] for _, argv in commands)
    sort = commands[1][1][4:]
    assert sort[:2] == ["sort", str(tmp_path / "s")]
    assert "--sorter" in sort and "mountainsort5" in sort and "--scratch-dir" in sort
    export = commands[2][1]
    assert export[export.index("--output-format") + 1] == "dh5" and "--no-lfp" in export and "--bids" in export
    assert export[export.index("--config") + 1] == str(tmp_path / "w" / "session.toml")


def test_work_dir_summary(tmp_path):
    (tmp_path / "trials").mkdir()
    pd.DataFrame({"start_time": [1.0, 2.0]}).to_csv(tmp_path / "trials" / "run_RF.csv", index=False)
    (tmp_path / "sorting" / "imec0").mkdir(parents=True)
    pd.DataFrame({"unit_id": [1, 2, 3], "KSLabel": ["good", "mua", "good"]}).to_csv(
        tmp_path / "sorting" / "imec0" / "units.csv", index=False)
    assert work_dir_summary(tmp_path, "s") == [("trials/run_RF.csv", "2 trials"), ("sorting/imec0", "3 units, 2 good")]


def test_load_edit_and_save_session_config(app, page, tmp_path):
    page.load_session()
    _wait(app, lambda: page._overview is not None)
    assert page._tree.topLevelItemCount() == 1
    run = page._tree.topLevelItem(0)
    assert run.text(0) == "3_rfmappingFine_g0"
    children = [run.child(i).text(1) for i in range(run.childCount())]
    assert any(c.startswith("imec0") for c in children) and any(c.startswith("stimstage: RF") for c in children)
    assert page._run_btn.isEnabled()

    config = page._editor.config()  # defaults from the recording: no session.toml yet
    assert [r.name for r in config.runs] == ["3_rfmappingFine_g0"] and set(config.probes) == {"imec0", "imec1"}
    config.probes["imec0"].area = "V1"
    config.probes["imec0"].bad_channels = [1]
    config.sorting.sorter = "mountainsort5"
    config.subject.species = "Mus musculus"
    page._editor.set_config(config)
    assert page.save_config()

    saved = load_session_config(tmp_path / "work" / "session.toml")
    assert saved.probes["imec0"].area == "V1" and saved.probes["imec0"].bad_channels == [1]
    assert saved.sorting.sorter == "mountainsort5" and saved.subject.species == "Mus musculus"
    assert saved.probes["imec1"].serial_number == config.probes["imec1"].serial_number


def test_save_as_and_open_config(app, page, tmp_path):
    page.load_session()
    _wait(app, lambda: page._overview is not None)
    config = page._editor.config()
    config.probes["imec0"].area = "LGN"
    page._editor.set_config(config)
    other = tmp_path / "templates" / "lgn.toml"
    assert page.save_config(other)
    assert page._config_path == other and load_session_config(other).probes["imec0"].area == "LGN"

    config.probes["imec0"].area = "changed"
    page._editor.set_config(config)
    page._reload_config()  # back to the file
    assert page._editor.config().probes["imec0"].area == "LGN"

    # a new page finds the config file last used for the session
    again = session_page.SessionPipelinePage()
    again._session_edit.setText(str(SPIKEGLX_CROP_SESSION))
    again._work_edit.setText(str(tmp_path / "work"))
    assert again._find_config() == other
    again.deleteLater()

    elsewhere = tmp_path / "v1.toml"
    config.probes["imec0"].area = "V1"
    save_session_config(config, elsewhere)
    assert page.open_config(elsewhere)
    assert page._config_path == elsewhere and page._editor.config().probes["imec0"].area == "V1"
    assert page.pipeline_options().config == elsewhere


def test_run_timing_step(app, page, tmp_path):
    page.load_session()
    _wait(app, lambda: page._overview is not None)
    for step, box in page._step_boxes.items():
        box.setChecked(step == "timing")
    page._run_or_cancel()
    assert page._process is not None
    _wait(app, lambda: page._process is None and page._run_btn.text().startswith("▶"), timeout_s=300)
    log = page._log.toPlainText()
    assert "all steps done" in log, log[-2000:]
    assert len(pd.read_csv(tmp_path / "work" / "trials" / "3_rfmappingFine_g0_RF.csv")) == 23
    outputs = [page._results.topLevelItem(i).text(0) for i in range(page._results.topLevelItemCount())]
    assert "trials/3_rfmappingFine_g0_RF.csv" in outputs
