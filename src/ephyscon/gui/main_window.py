import logging
import re
import sys
import threading
from pathlib import Path

from pydantic import ValidationError
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QComboBox, QSpinBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget, QProgressBar,
)


from ephyscon.config import (
    ContinuousMuaConfig, DecimationConfig, EventPreprocessingConfig,
    OpenEphysConversionConfig, OutputFormat, RawConfig,
    TrialMapConfig, load_config_from_file, save_config_to_file,
)
from ephyscon import convert_open_ephys_session
from ephyscon.convert_open_ephys_to_dh5 import get_output_paths
from ephyscon.version import get_version_from_pyproject
from ephyscon.inspect import validate_session_path

from ephyscon.gui.config_widget import ConfigStepWidget
from ephyscon.gui.inspector_widget import SessionInspectorWidget
from ephyscon.gui.session_page import SessionPipelinePage
from ephyscon.gui.settings import (
    get_last_config_path, get_last_session_dir,
    set_last_config_path, set_last_session_dir, pick_session_dirs,
)
from ephyscon.gui.widgets import ChannelPickerWidget

# (tab label, OpenEphysConversionConfig field name, model_class, enabled by default)
_TAB_CONFIGS = [
    ("Ra&w",       "raw_config",            RawConfig,                False),
    ("E&vents",    "event_config",          EventPreprocessingConfig, True),
    ("&Trial Map", "trialmap_config",       TrialMapConfig,           True),
    ("L&FP",       "decimation_config",     DecimationConfig,         True),
    ("M&UA",       "continuous_mua_config", ContinuousMuaConfig,      True),
]

_EVENTS_EXCLUDED = {"network_events_code_name_map", "ttl_line_names"}
_RAW_EXCLUDED = {"cont_ranges"}


def _format_validation_error(exc: ValidationError) -> str:
    model_name = exc.title
    lines = [f"{model_name} has invalid values:"]
    for error in exc.errors():
        field = " → ".join(str(part) for part in error["loc"])
        msg = re.sub(r"^value error,\s*", "", error["msg"], flags=re.IGNORECASE)
        got = error.get("input")
        got_str = f" (got {got!r})" if got is not None else ""
        lines.append(f"  • {field}: {msg}{got_str}")
    return "\n".join(lines)


class _QtLogHandler(logging.Handler):
    def __init__(self, signal: Signal):
        super().__init__()
        self._signal = signal
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._signal.emit(self.format(record))
        except Exception:
            pass


class _CancelledError(Exception):
    pass


class _ConversionWorker(QThread):
    log_message: Signal = Signal(str)
    step_progress: Signal = Signal(str, int, int)   # step_name, done, total
    session_progress: Signal = Signal(int, int)     # done, total
    succeeded: Signal = Signal()
    cancelled: Signal = Signal()
    error: Signal = Signal(str)
    conflict: Signal = Signal(list)                 # list of existing Path strings

    def __init__(
        self,
        session_paths: list[Path],
        output_folder: Path | None,
        config: OpenEphysConversionConfig,
        parent=None,
    ):
        super().__init__(parent)
        self._session_paths = session_paths
        self._output_folder = output_folder
        self._config = config
        self._conflict_event = threading.Event()
        self._conflict_action: str = "replace"
        self._replace_all = False
        self._skip_all = False

    def resolve_conflict(self, action: str) -> None:
        """Called from the main thread with 'replace', 'skip', or 'replace_all'."""
        self._conflict_action = action
        self._conflict_event.set()

    def run(self) -> None:
        handler = _QtLogHandler(self.log_message)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        n = len(self._session_paths)

        def on_progress(name: str, done: int, total: int) -> None:
            if self.isInterruptionRequested():
                raise _CancelledError()
            self.step_progress.emit(name, done, total)

        try:
            for i, session_path in enumerate(self._session_paths):
                if self.isInterruptionRequested():
                    raise _CancelledError()

                if not self._replace_all:
                    existing = [
                        str(p) for p in get_output_paths(
                            session_path, self._output_folder, self._config.output_format
                        )
                        if p.exists()
                    ]
                    if existing:
                        if self._skip_all:
                            self.session_progress.emit(i + 1, n)
                            continue
                        self._conflict_event.clear()
                        self.conflict.emit(existing)
                        self._conflict_event.wait()
                        if self._conflict_action in ("skip", "skip_all"):
                            if self._conflict_action == "skip_all":
                                self._skip_all = True
                            self.session_progress.emit(i + 1, n)
                            continue
                        if self._conflict_action == "replace_all":
                            self._replace_all = True

                self.session_progress.emit(i, n)
                convert_open_ephys_session(
                    session_path,
                    output_folder=self._output_folder,
                    config=self._config,
                    on_progress=on_progress,
                )
                self.session_progress.emit(i + 1, n)
            self.succeeded.emit()
        except _CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            root_logger.removeHandler(handler)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"ephyscon {get_version_from_pyproject()}")
        self.setMinimumSize(1000, 760)
        self._worker: _ConversionWorker | None = None
        self._setup_ui()
        self._restore_settings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        # page 1: session pipeline (SpikeGLX / Open Ephys sessions -> NWB/DH5/BIDS);
        # page 2: the Open Ephys -> DH5 conversion (`ephyscon convert`)
        self._pages = QTabWidget()
        self.setCentralWidget(self._pages)
        self._session_page = SessionPipelinePage()
        self._pages.addTab(self._session_page, "Session &pipeline")
        central = QWidget()
        self._pages.addTab(central, "Open Ephys → &DH5 conversion")
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # --- Input ---
        add_btn = QPushButton("&Add…")
        add_btn.clicked.connect(self._pick_session)
        remove_btn = QPushButton("Re&move")
        remove_btn.clicked.connect(self._remove_session)
        self._inspector = SessionInspectorWidget(buttons=[add_btn, remove_btn])
        self._inspector.setTitle("Input — Open Ephys Sessions")
        self._inspector.channels_changed.connect(self._on_channels_changed)
        root.addWidget(self._inspector)

        # --- Output ---
        output_group = QGroupBox("Output")
        output_form = QFormLayout(output_group)
        output_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Defaults to session parent folder")
        self._output_btn = QPushButton("&Browse")
        self._output_btn.clicked.connect(self._pick_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_edit)
        output_row.addWidget(self._output_btn)
        output_form.addRow("Folder:", output_row)

        meta_row = QHBoxLayout()
        self._format_combo = QComboBox()
        self._format_combo.addItem(OutputFormat.DH5.value.upper(), userData=OutputFormat.DH5)
        meta_row.addWidget(self._format_combo)
        meta_row.addSpacing(20)
        meta_row.addWidget(QLabel("Workers (n_jobs):"))
        self._n_jobs_spin = QSpinBox()
        self._n_jobs_spin.setRange(-1, 256)
        self._n_jobs_spin.setValue(1)
        self._n_jobs_spin.setToolTip("Parallelization not yet implemented.")
        self._n_jobs_spin.setEnabled(False)
        meta_row.addWidget(self._n_jobs_spin)
        meta_row.addStretch()
        output_form.addRow("Format:", meta_row)
        root.addWidget(output_group)

        # --- Config ---
        config_group = QGroupBox("Config")
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(6, 6, 6, 6)

        self._channel_pickers: dict[str, ChannelPickerWidget] = {
            "raw_config": ChannelPickerWidget(),
            "decimation_config": ChannelPickerWidget(),
            "continuous_mua_config": ChannelPickerWidget(),
        }

        self._tabs = QTabWidget()
        self._tab_widgets: dict[str, ConfigStepWidget] = {}
        for tab_name, field_name, model_class, enabled in _TAB_CONFIGS:
            excluded: set[str] = set()
            overrides: dict = {}
            if field_name == "raw_config":
                excluded = _RAW_EXCLUDED
                overrides = {"included_channel_names": self._channel_pickers["raw_config"]}
            elif field_name == "event_config":
                excluded = _EVENTS_EXCLUDED
            elif field_name == "decimation_config":
                overrides = {"included_channel_names": self._channel_pickers["decimation_config"]}
            elif field_name == "continuous_mua_config":
                overrides = {"included_channel_names": self._channel_pickers["continuous_mua_config"]}
            widget = ConfigStepWidget(
                model_class,
                enabled_by_default=enabled,
                excluded_fields=excluded,
                field_overrides=overrides,
            )
            self._tab_widgets[field_name] = widget
            self._tabs.addTab(widget, tab_name)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("&Load Config")
        load_btn.clicked.connect(self._load_config)
        save_btn = QPushButton("&Save Config")
        save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()

        config_layout.addWidget(self._tabs)
        config_layout.addLayout(btn_row)
        root.addWidget(config_group, stretch=3)

        # --- Progress ---
        progress_group = QGroupBox("Progress")
        progress_layout = QHBoxLayout(progress_group)
        progress_layout.setContentsMargins(6, 6, 6, 6)
        progress_layout.setSpacing(8)

        bars_layout = QVBoxLayout()
        label_width = 60

        session_lbl = QLabel("Sessions")
        session_lbl.setFixedWidth(label_width)
        self._session_bar = QProgressBar()
        self._session_bar.setTextVisible(True)
        self._session_bar.setRange(0, 0)
        self._session_bar.setValue(0)
        self._session_bar.setFormat("0/0")
        session_row = QHBoxLayout()
        session_row.addWidget(session_lbl)
        session_row.addWidget(self._session_bar, stretch=1)
        bars_layout.addLayout(session_row)

        self._step_label = QLabel("Steps")
        self._step_label.setFixedWidth(label_width)
        self._step_bar = QProgressBar()
        self._step_bar.setTextVisible(True)
        self._step_bar.setRange(0, 0)
        self._step_bar.setValue(0)
        self._step_bar.setFormat("—")
        step_row = QHBoxLayout()
        step_row.addWidget(self._step_label)
        step_row.addWidget(self._step_bar, stretch=1)
        bars_layout.addLayout(step_row)

        self._run_btn = QPushButton("▶  &Run")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._run)

        progress_layout.addLayout(bars_layout, stretch=1)
        progress_layout.addWidget(self._run_btn)
        root.addWidget(progress_group)

        # --- Log ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 6, 6, 6)
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFixedHeight(140)
        mono = QFont("Courier New" if sys.platform == "win32" else "Monospace")
        mono.setPointSize(9)
        self._log_edit.setFont(mono)
        log_layout.addWidget(self._log_edit)
        root.addWidget(log_group)

    # ------------------------------------------------------------------
    # Path pickers
    # ------------------------------------------------------------------

    def _on_channels_changed(self) -> None:
        channels = self._inspector.all_channel_names()
        for picker in self._channel_pickers.values():
            picker.set_available_channels(channels)

    def _pick_session(self) -> None:
        paths = pick_session_dirs(self, initial=get_last_session_dir() or "")
        for p in paths:
            try:
                validate_session_path(p)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid session", str(exc))
                continue
            set_last_session_dir(p)
            if not self._output_edit.text():
                self._output_edit.setText(str(p.parent))
            self._inspector.add(p)
        self._update_session_bar()

    def _remove_session(self) -> None:
        self._inspector.remove_selected()
        self._update_session_bar()

    def _update_session_bar(self) -> None:
        n = len(self._inspector.session_paths())
        self._session_bar.setRange(0, max(n, 1))
        self._session_bar.setValue(0)
        self._session_bar.setFormat(f"0/{n}")

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._output_edit.setText(path)

    # ------------------------------------------------------------------
    # Config build / load / save
    # ------------------------------------------------------------------

    def _build_config(self) -> OpenEphysConversionConfig:
        kwargs: dict = {
            field_name: self._tab_widgets[field_name].get_model()
            for _, field_name, _, _ in _TAB_CONFIGS
        }
        kwargs["output_format"] = self._format_combo.currentData()
        kwargs["n_jobs"] = self._n_jobs_spin.value()
        return OpenEphysConversionConfig(**kwargs)

    def _apply_config(self, config: OpenEphysConversionConfig) -> None:
        for _, field_name, _, _ in _TAB_CONFIGS:
            self._tab_widgets[field_name].set_model(getattr(config, field_name, None))
        for i in range(self._format_combo.count()):
            if self._format_combo.itemData(i) == config.output_format:
                self._format_combo.setCurrentIndex(i)
                break
        self._n_jobs_spin.setValue(config.n_jobs)

    def _restore_settings(self) -> None:
        config_path = get_last_config_path()
        if config_path:
            try:
                self._apply_config(load_config_from_file(config_path))
            except Exception:
                pass  # silently ignore stale/broken config

    def _load_config(self) -> None:
        initial = get_last_config_path() or ""
        path, _ = QFileDialog.getOpenFileName(self, "Load config", initial, filter="JSON (*.json)")
        if not path:
            return
        try:
            self._apply_config(load_config_from_file(path))
            set_last_config_path(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _save_config(self) -> None:
        initial = get_last_config_path() or ""
        path, _ = QFileDialog.getSaveFileName(self, "Save config", initial, filter="JSON (*.json)")
        if not path:
            return
        try:
            save_config_to_file(path, self._build_config())
            set_last_config_path(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self) -> None:
        session_paths = self._inspector.session_paths()
        if not session_paths:
            QMessageBox.warning(self, "No session", "Please add at least one session folder.")
            return

        output_str = self._output_edit.text().strip()
        output_folder = Path(output_str) if output_str else None

        try:
            config = self._build_config()
        except ValidationError as exc:
            QMessageBox.critical(self, "Config error", _format_validation_error(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Config error", str(exc))
            return

        self._log_edit.clear()
        self._set_inputs_enabled(False)
        self._set_run_mode(True)

        n = len(session_paths)
        self._step_bar.setRange(0, 1)
        self._step_bar.setValue(0)
        self._step_bar.setFormat("—")
        self._session_bar.setRange(0, n)
        self._session_bar.setValue(0)
        self._session_bar.setFormat(f"0/{n}")

        self._worker = _ConversionWorker(session_paths, output_folder, config, parent=self)
        self._worker.log_message.connect(self._append_log)
        self._worker.step_progress.connect(self._on_step_progress)
        self._worker.session_progress.connect(self._on_session_progress)
        self._worker.succeeded.connect(self._on_finished)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.error.connect(self._on_error)
        self._worker.conflict.connect(self._on_conflict)
        self._worker.start()

    def _append_log(self, msg: str) -> None:
        self._log_edit.append(msg)

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self._inspector.setEnabled(enabled)
        self._output_edit.setEnabled(enabled)
        self._output_btn.setEnabled(enabled)
        self._format_combo.setEnabled(enabled)
        self._tabs.setEnabled(enabled)

    def _set_run_mode(self, running: bool) -> None:
        if running:
            self._run_btn.setText("■  &Cancel")
            self._run_btn.clicked.disconnect(self._run)
            self._run_btn.clicked.connect(self._cancel)
        else:
            self._run_btn.setText("▶  &Run")
            self._run_btn.setEnabled(True)
            self._run_btn.clicked.disconnect(self._cancel)
            self._run_btn.clicked.connect(self._run)

    def _cancel(self) -> None:
        if self._worker:
            self._worker.requestInterruption()
            self._run_btn.setEnabled(False)
            self._run_btn.setText("Cancelling…")

    def _on_step_progress(self, step_name: str, done: int, total: int) -> None:
        self._step_bar.setRange(0, total)
        self._step_bar.setValue(done)
        self._step_bar.setFormat(f"{step_name}  {done}/{total}")

    def _on_session_progress(self, done: int, total: int) -> None:
        self._session_bar.setValue(done)
        self._session_bar.setFormat(f"{done}/{total}")

    def _on_conflict(self, existing: list) -> None:
        names = "\n".join(f"  • {Path(p).name}" for p in existing)
        msg = QMessageBox(self)
        msg.setWindowTitle("Output file(s) already exist")
        msg.setText(f"The following file(s) will be overwritten:\n\n{names}")
        replace_btn = msg.addButton("&Replace this session", QMessageBox.ButtonRole.AcceptRole)
        skip_btn = msg.addButton("&Skip this session", QMessageBox.ButtonRole.RejectRole)
        replace_all_btn = msg.addButton("Replace &all", QMessageBox.ButtonRole.AcceptRole)
        skip_all_btn = msg.addButton("Skip all e&xisting", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is replace_all_btn:
            action = "replace_all"
        elif clicked is skip_all_btn:
            action = "skip_all"
        elif clicked is skip_btn:
            action = "skip"
        else:
            action = "replace"
        if self._worker:
            self._worker.resolve_conflict(action)

    def _on_finished(self) -> None:
        self._set_run_mode(False)
        self._set_inputs_enabled(True)
        self._step_bar.setFormat("—")
        self._append_log("--- Conversion complete ---")

    def _on_cancelled(self) -> None:
        self._set_run_mode(False)
        self._set_inputs_enabled(True)
        self._step_bar.setFormat("—")
        self._append_log("--- Conversion cancelled ---")

    def _on_error(self, msg: str) -> None:
        self._set_run_mode(False)
        self._set_inputs_enabled(True)
        self._step_bar.setFormat("—")
        self._append_log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Conversion failed", msg)
