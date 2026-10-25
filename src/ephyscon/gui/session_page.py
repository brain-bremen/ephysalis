"""GUI page of the session pipeline: pick a session, edit its ``session.toml``, run the
pipeline steps (each one as an ``ephyscon`` process, see :mod:`ephyscon.gui.pipeline`) and
see what they wrote to the work dir."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from pydantic import ValidationError
from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ephyscon.gui.config_widget import ConfigStepWidget
from ephyscon.gui.pipeline import STEPS, PipelineOptions, step_commands, work_dir_summary
from ephyscon.session_config import (
    SESSION_CONFIG_FILENAME, AuxChannelMap, ProbeConfig, RunConfig, SessionConfig, SortingConfig,
    SubjectConfig, default_session_config, load_session_config, save_session_config,
)

_GENERAL_EXCLUDED = {"schema_version", "runs", "probes", "subject", "sorting", "aux_channels"}
_AUTO = "(auto-detect)"
COMMON_SORTERS = ["kilosort4", "mountainsort5", "spykingcircus2", "tridesclous2"]


_TOML_FILTER = "Session config (*.toml);;All files (*)"


def _settings() -> QSettings:
    return QSettings("brain-bremen", "ephyscon")


def _config_key(session: Path) -> str:
    """QSettings key of the config file last used for ``session``."""
    return "pipeline/config/" + str(session.resolve()).replace("\\", "/").replace("/", "|")


def _format_validation_error(exc: ValidationError) -> str:
    lines = [f"{exc.title} has invalid values:"]
    for error in exc.errors():
        lines.append(f"  • {' → '.join(str(p) for p in error['loc'])}: {error['msg']}")
    return "\n".join(lines)


def _path_row(edit: QLineEdit, on_browse) -> QHBoxLayout:
    button = QPushButton("Browse…")
    button.clicked.connect(on_browse)
    row = QHBoxLayout()
    row.addWidget(edit)
    row.addWidget(button)
    return row


# ---------------------------------------------------------------------------
# Session overview (loaded in a thread: discovery reads metadata over the network)
# ---------------------------------------------------------------------------

@dataclass
class SessionOverview:
    name: str
    format: str
    runs: list[dict] = field(default_factory=list)  # name, start, duration, probes, aux, paradigms
    probe_ids: list[str] = field(default_factory=list)
    default_config: SessionConfig | None = None


class _OverviewLoader(QThread):
    loaded: Signal = Signal(object)
    failed: Signal = Signal(str)

    def __init__(self, session_path: Path, config: SessionConfig | None, config_dir: Path | None, parent=None):
        super().__init__(parent)
        self._session_path, self._config, self._config_dir = session_path, config, config_dir

    def run(self) -> None:
        try:
            from ephyscon.io.base import open_session
            from ephyscon.io.probes import configure_probes
            from ephyscon.stimulus.base import detect_stimulus_system, get_stimulus_system

            session = open_session(self._session_path)
            overview = SessionOverview(session.name, session.format.name, default_config=default_session_config(session))
            configure_probes(session, self._config, self._config_dir)
            overview.probe_ids = session.probe_ids
            for run in session.runs:
                run_config = self._config.run_config(run.name) if self._config else None
                try:
                    system = (get_stimulus_system(run_config.stimulus_system)
                              if run_config and run_config.stimulus_system else detect_stimulus_system(run))
                    paradigms = [f"{system.name}: {p.name}" for p in system.discover(run)] if system else []
                except Exception as exc:  # a broken stimulus folder must not hide the session
                    paradigms = [f"error: {exc}"]
                overview.runs.append({
                    "name": run.name,
                    "start": run.start_time.isoformat(sep=" ", timespec="seconds") if run.start_time else "?",
                    "duration": f"{run.duration_s:.0f} s" if run.duration_s else "?",
                    "probes": [f"{p.probe_id}: {p.part_number or p.model_name or '?'}, {p.num_channels or '?'} ch"
                               for p in run.probes.values()],
                    "aux": run.aux_stream or "",
                    "paradigms": paradigms,
                })
            self.loaded.emit(overview)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# session.toml editor
# ---------------------------------------------------------------------------

class _RunsTable(QTableWidget):
    """Runs of the session: include, stimulus system, notes."""

    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["Run", "Include", "Stimulus system", "Notes"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

    def set_runs(self, runs: list[RunConfig]) -> None:
        from ephyscon.plugins import STIMULUS_SYSTEM_GROUP, available

        systems = available(STIMULUS_SYSTEM_GROUP)
        self.setRowCount(0)
        for run in runs:
            row = self.rowCount()
            self.insertRow(row)
            name = QTableWidgetItem(run.name)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, name)
            include = QCheckBox()
            include.setChecked(run.include)
            self.setCellWidget(row, 1, include)
            combo = QComboBox()
            combo.addItems([_AUTO, *systems])
            if run.stimulus_system:
                combo.setCurrentText(run.stimulus_system)
            self.setCellWidget(row, 2, combo)
            self.setItem(row, 3, QTableWidgetItem(run.notes))

    def runs(self) -> list[RunConfig]:
        result = []
        for row in range(self.rowCount()):
            system = cast(QComboBox, self.cellWidget(row, 2)).currentText()
            notes = self.item(row, 3)
            result.append(RunConfig(
                name=cast(QTableWidgetItem, self.item(row, 0)).text(),
                include=cast(QCheckBox, self.cellWidget(row, 1)).isChecked(),
                stimulus_system=None if system == _AUTO else system,
                notes=notes.text() if notes is not None else "",
            ))
        return result


class _ProbesEditor(QWidget):
    """Probe list with the form of the selected probe; probes not detected by the reader
    (non-Neuropixels) are added with a name and a ``stream``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._probes: dict[str, ProbeConfig] = {}
        self._current: str | None = None
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        self._list = QListWidget()
        self._list.currentTextChanged.connect(self._select)
        left.addWidget(self._list)
        buttons = QHBoxLayout()
        add = QPushButton("Add…")
        add.setToolTip("Add a probe on a stream the reader does not detect (set its stream, channels and geometry)")
        add.clicked.connect(self._add)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        left.addLayout(buttons)
        layout.addLayout(left, 1)
        self._form = ConfigStepWidget(ProbeConfig, checkable=False)
        layout.addWidget(self._form, 3)

    def _store_current(self) -> None:
        if self._current is not None and self._current in self._probes:
            self._probes[self._current] = self._form.get_model()  # type: ignore[assignment]

    def _select(self, probe_id: str) -> None:
        self._store_current()
        self._current = probe_id or None
        if self._current in self._probes:
            self._form.set_model(self._probes[self._current])

    def _add(self) -> None:
        name, ok = QInputDialog.getText(self, "Add probe", "Probe id (e.g. probe0):")
        if ok and name and name not in self._probes:
            self._store_current()
            self._probes[name] = ProbeConfig()
            self._list.addItem(name)
            self._list.setCurrentRow(self._list.count() - 1)

    def _remove(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            self._probes.pop(item.text(), None)
            self._current = None
            self._list.takeItem(self._list.row(item))

    def set_probes(self, probes: dict[str, ProbeConfig]) -> None:
        self._current = None
        self._probes = {k: v.model_copy(deep=True) for k, v in probes.items()}
        self._list.clear()
        self._list.addItems(list(self._probes))
        if self._probes:
            self._list.setCurrentRow(0)

    def probes(self) -> dict[str, ProbeConfig]:
        self._store_current()
        return dict(self._probes)


class SessionConfigEditor(QTabWidget):
    """All sections of ``session.toml``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._general = ConfigStepWidget(SessionConfig, checkable=False, excluded_fields=_GENERAL_EXCLUDED)
        self._subject = ConfigStepWidget(SubjectConfig, checkable=False)
        self._runs = _RunsTable()
        self._probes = _ProbesEditor()
        self._sorting = ConfigStepWidget(SortingConfig, checkable=False)
        self._sorting.setToolTip(f"Any spikeinterface sorter, e.g. {', '.join(COMMON_SORTERS)}")
        self._aux = ConfigStepWidget(AuxChannelMap, checkable=False)
        for widget, label in ((self._general, "&General"), (self._subject, "S&ubject"), (self._runs, "R&uns"),
                              (self._probes, "&Probes"), (self._sorting, "S&orting"), (self._aux, "&Aux channels")):
            self.addTab(widget, label)
        self._base: SessionConfig | None = None

    def set_config(self, config: SessionConfig) -> None:
        self._base = config
        self._general.set_field_values(config)
        self._subject.set_model(config.subject)
        self._runs.set_runs(config.runs)
        self._probes.set_probes(config.probes)
        self._sorting.set_model(config.sorting)
        self._aux.set_model(config.aux_channels)

    def config(self) -> SessionConfig:
        """The edited config (raises ValidationError or ValueError for invalid input)."""
        general = {name: value for name, value in self._general.get_model().model_dump().items()  # type: ignore[union-attr]
                   if name not in _GENERAL_EXCLUDED}
        base = self._base.model_dump() if self._base is not None else {}
        return SessionConfig.model_validate({
            **base, **general,
            "subject": self._subject.get_model(), "runs": self._runs.runs(), "probes": self._probes.probes(),
            "sorting": self._sorting.get_model(), "aux_channels": self._aux.get_model(),
        })


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class SessionPipelinePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._config_path: Path | None = None
        self._loader: _OverviewLoader | None = None
        self._overview: SessionOverview | None = None
        self._process: QProcess | None = None
        self._queue: list[tuple[str, list[str]]] = []
        self._n_steps = 0
        self._cancelled = False
        self._setup_ui()
        last = _settings().value("pipeline/session")
        if last and Path(str(last)).is_dir():
            self._session_edit.setText(str(last))
            work = _settings().value("pipeline/work_dir")
            self._work_edit.setText(str(work) if work else "")

    # --- UI -------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)

        session_group = QGroupBox("Session")
        form = QFormLayout(session_group)
        self._session_edit = QLineEdit()
        self._session_edit.setPlaceholderText("SpikeGLX or Open Ephys session folder")
        form.addRow("Session:", _path_row(self._session_edit, self._browse_session))
        self._work_edit = QLineEdit()
        self._work_edit.setPlaceholderText("default: <session>/ephyscon")
        form.addRow("Work dir:", _path_row(self._work_edit, self._browse_work_dir))
        load_row = QHBoxLayout()
        load = QPushButton("&Load session")
        load.clicked.connect(self.load_session)
        self._status = QLabel("")
        load_row.addWidget(load)
        load_row.addWidget(self._status, 1)
        form.addRow(load_row)
        root.addWidget(session_group)

        splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        overview_group = QGroupBox("Runs, probes and paradigms")
        overview_layout = QVBoxLayout(overview_group)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Item", "Details"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        overview_layout.addWidget(self._tree)
        left_layout.addWidget(overview_group, 2)

        results_group = QGroupBox("Outputs in the work dir")
        results_layout = QVBoxLayout(results_group)
        self._results = QTreeWidget()
        self._results.setHeaderLabels(["Output", "Content"])
        self._results.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        results_layout.addWidget(self._results)
        open_row = QHBoxLayout()
        open_work = QPushButton("Open &work dir")
        open_work.clicked.connect(lambda: self._open_folder(self.work_dir()))
        self._open_bids = QPushButton("Open &BIDS dataset")
        self._open_bids.clicked.connect(self._open_bids_dataset)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_results)
        open_row.addWidget(open_work)
        open_row.addWidget(self._open_bids)
        open_row.addWidget(refresh)
        open_row.addStretch()
        results_layout.addLayout(open_row)
        left_layout.addWidget(results_group, 1)
        splitter.addWidget(left)

        config_group = QGroupBox(SESSION_CONFIG_FILENAME)
        config_layout = QVBoxLayout(config_group)
        self._config_label = QLabel("no session loaded")
        self._config_label.setWordWrap(True)
        config_layout.addWidget(self._config_label)
        self._editor = SessionConfigEditor()
        self._editor.setEnabled(False)
        config_layout.addWidget(self._editor)
        buttons = QHBoxLayout()
        self._open_btn = QPushButton("&Open…")
        self._open_btn.setToolTip("Use another config file for this session (e.g. one of a similar session as a template)")
        self._open_btn.clicked.connect(self._open_config_dialog)
        self._save_btn = QPushButton("&Save")
        self._save_btn.clicked.connect(self.save_config)
        self._save_as_btn = QPushButton("Save &as…")
        self._save_as_btn.clicked.connect(self._save_config_as_dialog)
        self._reload_btn = QPushButton("Re&load")
        self._reload_btn.setToolTip("Discard the changes: read the config file again")
        self._reload_btn.clicked.connect(self._reload_config)
        self._defaults_btn = QPushButton("Reset to &defaults")
        self._defaults_btn.setToolTip("Config from the recording files (all runs included, probes from the metadata)")
        self._defaults_btn.clicked.connect(self._reset_to_defaults)
        for b in (self._open_btn, self._save_btn, self._save_as_btn, self._reload_btn, self._defaults_btn):
            b.setEnabled(False)
            buttons.addWidget(b)
        buttons.addStretch()
        config_layout.addLayout(buttons)
        splitter.addWidget(config_group)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 3)

        steps_group = QGroupBox("Steps")
        steps_layout = QVBoxLayout(steps_group)
        boxes = QHBoxLayout()
        self._step_boxes: dict[str, QCheckBox] = {}
        for step, label, description in STEPS:
            box = QCheckBox(label)
            box.setChecked(True)
            box.setToolTip(description)
            self._step_boxes[step] = box
            boxes.addWidget(box)
        boxes.addStretch()
        steps_layout.addLayout(boxes)

        options = QFormLayout()
        jobs_row = QHBoxLayout()
        self._jobs_spin = QSpinBox()
        self._jobs_spin.setRange(1, 64)
        self._jobs_spin.setValue(4)
        jobs_row.addWidget(self._jobs_spin)
        jobs_row.addWidget(QLabel("Sorter:"))
        self._sorter_combo = QComboBox()
        self._sorter_combo.setEditable(True)
        self._sorter_combo.addItems(["(from session.toml)", *COMMON_SORTERS])
        jobs_row.addWidget(self._sorter_combo)
        jobs_row.addStretch()
        options.addRow("Parallel jobs:", jobs_row)
        self._scratch_edit = QLineEdit(str(_settings().value("pipeline/scratch_dir") or ""))
        self._scratch_edit.setPlaceholderText("fast local disk for the sorting copy (default: system temp)")
        options.addRow("Scratch dir:", _path_row(self._scratch_edit, self._browse_scratch))
        export_row = QHBoxLayout()
        self._format_combo = QComboBox()
        self._format_combo.addItems(["nwb", "dh5"])
        export_row.addWidget(self._format_combo)
        self._export_lfp = QCheckBox("with LFP")
        self._export_lfp.setChecked(True)
        export_row.addWidget(self._export_lfp)
        export_row.addWidget(QLabel("BIDS dataset:"))
        self._bids_edit = QLineEdit(str(_settings().value("pipeline/bids_root") or ""))
        self._bids_edit.setPlaceholderText("empty: no BIDS (DH5 gives a BIDS-like layout)")
        export_row.addLayout(_path_row(self._bids_edit, self._browse_bids), 1)
        options.addRow("Export:", export_row)
        steps_layout.addLayout(options)

        run_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setFormat("—")
        self._progress.setTextVisible(True)
        self._run_btn = QPushButton("▶  &Run steps")
        self._run_btn.clicked.connect(self._run_or_cancel)
        self._run_btn.setEnabled(False)
        run_row.addWidget(self._progress, 1)
        run_row.addWidget(self._run_btn)
        steps_layout.addLayout(run_row)
        root.addWidget(steps_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        mono = QFont("Courier New" if sys.platform == "win32" else "Monospace")
        mono.setPointSize(9)
        self._log.setFont(mono)
        log_layout.addWidget(self._log)
        root.addWidget(log_group, 1)

    # --- paths ----------------------------------------------------------

    def session_path(self) -> Path | None:
        text = self._session_edit.text().strip()
        return Path(text) if text else None

    def work_dir(self) -> Path:
        text = self._work_edit.text().strip()
        if text:
            return Path(text)
        from ephyscon.timing import WORK_DIR_NAME

        session = self.session_path()
        return (session / WORK_DIR_NAME) if session else Path(".")

    def _browse(self, edit: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title, edit.text() or str(self.session_path() or ""))
        if path:
            edit.setText(path)

    def _browse_session(self) -> None:
        self._browse(self._session_edit, "Session folder")

    def _browse_work_dir(self) -> None:
        self._browse(self._work_edit, "Work dir")

    def _browse_scratch(self) -> None:
        self._browse(self._scratch_edit, "Scratch dir (fast local disk)")

    def _browse_bids(self) -> None:
        self._browse(self._bids_edit, "BIDS dataset folder")

    def _open_bids_dataset(self) -> None:
        self._open_folder(Path(self._bids_edit.text()))

    @staticmethod
    def _open_folder(path: Path) -> None:
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # --- session and config ---------------------------------------------

    def _find_config(self) -> Path | None:
        """The config file last opened or saved for this session, else ``session.toml`` in
        the session folder or the work dir."""
        session = self.session_path()
        candidates = [self.work_dir() / SESSION_CONFIG_FILENAME]
        if session is not None:
            candidates.insert(0, session / SESSION_CONFIG_FILENAME)
            remembered = _settings().value(_config_key(session))
            if remembered:
                candidates.insert(0, Path(str(remembered)))
        return next((p for p in candidates if p.exists()), None)

    def _set_config_path(self, path: Path) -> None:
        self._config_path = path
        self._config_label.setText(str(path))
        session = self.session_path()
        if session is not None:
            _settings().setValue(_config_key(session), str(path))

    def load_session(self) -> None:
        session = self.session_path()
        if session is None or not session.is_dir():
            QMessageBox.warning(self, "No session", "Please choose a session folder.")
            return
        _settings().setValue("pipeline/session", str(session))
        _settings().setValue("pipeline/work_dir", self._work_edit.text().strip())
        self._config_path = self._find_config()
        config = None
        if self._config_path is not None:
            try:
                config = load_session_config(self._config_path)
            except Exception as exc:
                QMessageBox.critical(self, "Invalid session.toml", f"{self._config_path}:\n{exc}")
                return
        self._status.setText("loading…")
        self._tree.clear()
        self._loader = _OverviewLoader(session, config, self._config_path.parent if self._config_path else None, self)
        self._loader.loaded.connect(lambda overview: self._on_loaded(overview, config))
        self._loader.failed.connect(self._on_load_failed)
        self._loader.start()

    def _on_load_failed(self, message: str) -> None:
        self._status.setText("")
        QMessageBox.critical(self, "Cannot open session", message)

    def _on_loaded(self, overview: SessionOverview, config: SessionConfig | None) -> None:
        self._overview = overview
        self._status.setText(f"{overview.name} ({overview.format}), {len(overview.runs)} runs")
        for run in overview.runs:
            item = QTreeWidgetItem([run["name"], f"start {run['start']}, {run['duration']}"])
            for probe in run["probes"]:
                item.addChild(QTreeWidgetItem(["probe", probe]))
            if run["aux"]:
                item.addChild(QTreeWidgetItem(["aux", run["aux"]]))
            for paradigm in run["paradigms"] or ["none found"]:
                item.addChild(QTreeWidgetItem(["paradigm", paradigm]))
            self._tree.addTopLevelItem(item)
            item.setExpanded(True)
        if config is None:
            config = overview.default_config
            self._config_label.setText(
                f"No {SESSION_CONFIG_FILENAME} yet: showing the defaults; Save writes "
                f"{self.work_dir() / SESSION_CONFIG_FILENAME}")
        else:
            self._config_label.setText(str(self._config_path))
        assert config is not None
        self._editor.set_config(config)
        self._editor.setEnabled(True)
        for b in (self._open_btn, self._save_btn, self._save_as_btn, self._reload_btn, self._defaults_btn,
                  self._run_btn):
            b.setEnabled(True)
        self.refresh_results()

    def save_config(self, path: Path | None = None) -> bool:
        """Save the editor to ``path`` (default: the current config file, else
        ``<work dir>/session.toml``), which becomes the config file of the session."""
        try:
            config = self._editor.config()
        except ValidationError as exc:
            QMessageBox.critical(self, "Invalid config", _format_validation_error(exc))
            return False
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid config", str(exc))
            return False
        path = path or self._config_path or (self.work_dir() / SESSION_CONFIG_FILENAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_session_config(config, path)
        self._set_config_path(path)
        self._append(f"Saved {path}")
        return True

    def open_config(self, path: Path) -> bool:
        """Show the config in ``path``, which becomes the config file of the session."""
        try:
            config = load_session_config(path)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid config", f"{path}:\n{exc}")
            return False
        self._editor.set_config(config)
        self._set_config_path(path)
        self._append(f"Opened {path}")
        known = {run["name"] for run in self._overview.runs} if self._overview else set()
        unknown = [run.name for run in config.runs if known and run.name not in known]
        if unknown:
            QMessageBox.warning(self, "Runs not in this session",
                                "The config lists runs that this session does not have:\n" + "\n".join(unknown))
        return True

    def _config_dialog_dir(self) -> str:
        return str(self._config_path.parent if self._config_path else self.work_dir())

    def _open_config_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open session config", self._config_dialog_dir(), _TOML_FILTER)
        if path:
            self.open_config(Path(path))

    def _save_config_as_dialog(self) -> None:
        initial = self._config_path or (self.work_dir() / SESSION_CONFIG_FILENAME)
        path, _ = QFileDialog.getSaveFileName(self, "Save session config", str(initial), _TOML_FILTER)
        if path:
            self.save_config(Path(path))

    def _reload_config(self) -> None:
        if self._config_path is not None and self._config_path.exists():
            self.open_config(self._config_path)

    def _reset_to_defaults(self) -> None:
        if self._overview is not None and self._overview.default_config is not None:
            self._editor.set_config(self._overview.default_config)

    def refresh_results(self) -> None:
        self._results.clear()
        name = self._overview.name if self._overview else ""
        for output, description in work_dir_summary(self.work_dir(), name):
            self._results.addTopLevelItem(QTreeWidgetItem([output, description]))

    # --- running the steps ----------------------------------------------

    def pipeline_options(self) -> PipelineOptions:
        session = self.session_path()
        assert session is not None
        sorter = self._sorter_combo.currentText().strip()
        return PipelineOptions(
            session=session, work_dir=self.work_dir(), config=self._config_path,
            steps=[s for s, box in self._step_boxes.items() if box.isChecked()],
            n_jobs=self._jobs_spin.value(),
            scratch_dir=Path(self._scratch_edit.text()) if self._scratch_edit.text().strip() else None,
            sorter=None if sorter.startswith("(") or not sorter else sorter,
            output_format=self._format_combo.currentText(),
            bids_root=Path(self._bids_edit.text()) if self._bids_edit.text().strip() else None,
            export_lfp=self._export_lfp.isChecked(),
        )

    def _run_or_cancel(self) -> None:
        if self._process is not None:
            self._cancelled = True
            self._queue.clear()
            self._process.kill()
            return
        if not self.save_config():
            return
        options = self.pipeline_options()
        _settings().setValue("pipeline/scratch_dir", self._scratch_edit.text().strip())
        _settings().setValue("pipeline/bids_root", self._bids_edit.text().strip())
        self._queue = step_commands(options)
        if not self._queue:
            QMessageBox.information(self, "No steps", "Please choose at least one step.")
            return
        self._n_steps = len(self._queue)
        self._cancelled = False
        self._progress.setRange(0, self._n_steps)
        self._progress.setValue(0)
        self._editor.setEnabled(False)
        self._run_btn.setText("■  &Cancel")
        self._start_next()

    def _start_next(self) -> None:
        if not self._queue:
            self._finish("--- all steps done ---")
            return
        step, argv = self._queue.pop(0)
        done = self._n_steps - len(self._queue) - 1
        self._progress.setValue(done)
        self._progress.setFormat(f"{step}  ({done + 1}/{self._n_steps})")
        self._append(f"$ ephyscon {' '.join(argv[4:])}")
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(lambda code, status, step=step: self._step_finished(step, code, status))
        self._process = process
        process.start(argv[0], argv[1:])

    def _read_output(self) -> None:
        if self._process is not None:
            text = bytes(self._process.readAllStandardOutput().data()).decode("utf-8", errors="replace")
            for line in text.rstrip("\n").splitlines():
                self._append(line)

    def _step_finished(self, step: str, code: int, status: QProcess.ExitStatus) -> None:
        self._read_output()
        self._process = None
        self.refresh_results()
        if self._cancelled:
            self._finish(f"--- cancelled during {step} (spike sorting resumes from its checkpoints) ---", error=True)
        elif status != QProcess.ExitStatus.NormalExit or code != 0:
            self._finish(f"--- {step} failed (exit code {code}) ---", error=True)
        else:
            self._start_next()

    def _finish(self, message: str, error: bool = False) -> None:
        self._queue.clear()
        self._process = None
        self._append(message)
        self._progress.setValue(self._n_steps if not error else self._progress.value())
        self._progress.setFormat("—" if error else "done")
        self._editor.setEnabled(True)
        self._run_btn.setText("▶  &Run steps")

    def _append(self, line: str) -> None:
        self._log.append(line)
