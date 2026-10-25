"""Main window of ``ephyscon-gui``: the converter page (:mod:`ephyscon.gui.session_page`)."""

from __future__ import annotations

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox

from ephyscon.gui.session_page import SessionPipelinePage
from ephyscon.version import get_version_from_pyproject


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"ephyscon {get_version_from_pyproject()}")
        self.setMinimumSize(1000, 760)
        self._page = SessionPipelinePage()
        self.setCentralWidget(self._page)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._page._process is not None:
            answer = QMessageBox.question(self, "Steps running", "Steps are still running. Cancel them and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._page._run_or_cancel()
        if not self._page._confirm_discard():
            event.ignore()
            return
        event.accept()
