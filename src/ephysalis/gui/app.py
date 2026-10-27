import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from ephysalis.gui.main_window import MainWindow


def _apply_dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(160, 160, 160))
    disabled = QPalette.ColorGroup.Disabled
    palette.setColor(disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    palette.setColor(disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    app.setPalette(palette)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("ephysalis")
    _apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
