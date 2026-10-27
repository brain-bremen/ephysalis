import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from ephysalis.gui.app import _apply_dark_palette
from ephysalis.version import get_version_from_pyproject
from ephysalis.gui.inspector_widget import SessionInspectorWidget
from ephysalis.gui.settings import get_last_session_dir, set_last_session_dir, pick_session_dirs
from ephysalis.inspect import validate_session_path


class _InspectorWindow(QMainWindow):
    def __init__(self, session_path: Path | None = None):
        super().__init__()
        self.setWindowTitle(f"ephysalis {get_version_from_pyproject()} — Session Inspector")
        self.setMinimumSize(520, 400)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        add_btn = QPushButton("Add…")
        add_btn.clicked.connect(self._pick_session)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_session)
        self._inspector = SessionInspectorWidget(buttons=[add_btn, remove_btn])
        layout.addWidget(self._inspector, stretch=1)

        if session_path is not None:
            self._inspector.add(session_path)

    def _pick_session(self) -> None:
        paths = pick_session_dirs(self, initial=get_last_session_dir() or "")
        for p in paths:
            try:
                validate_session_path(p)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid session", str(exc))
                continue
            set_last_session_dir(p)
            self._inspector.add(p)

    def _remove_session(self) -> None:
        self._inspector.remove_selected()


def main() -> None:
    session_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    app = QApplication(sys.argv)
    app.setApplicationName("ephysalis inspector")
    _apply_dark_palette(app)

    window = _InspectorWindow(session_path)
    window.show()
    sys.exit(app.exec())
