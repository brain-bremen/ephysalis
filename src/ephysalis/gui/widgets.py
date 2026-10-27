from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QListWidget, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QVBoxLayout, QWidget,
)


class ListEditor(QWidget):
    """Editable list[str] — one entry per line. Empty text → None."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._edit = QTextEdit()
        self._edit.setFixedHeight(80)
        self._edit.setPlaceholderText("One entry per line (empty = include all)")
        layout.addWidget(self._edit)

    def get_value(self) -> list[str] | None:
        text = self._edit.toPlainText().strip()
        if not text:
            return None
        return [line.strip() for line in text.splitlines() if line.strip()]

    def set_value(self, value: list[str] | None) -> None:
        self._edit.setPlainText("\n".join(value) if value else "")


class DictEditor(QWidget):
    """Editable dict[str, int] — a table with Key and Value columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Key", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setFixedHeight(120)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+")
        add_btn.setFixedWidth(30)
        add_btn.clicked.connect(self._add_row)
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.clicked.connect(self._remove_row)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _add_row(self) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(""))
        self._table.setItem(row, 1, QTableWidgetItem("0"))

    def _remove_row(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)

    def get_value(self) -> dict[str, int] | None:
        result: dict[str, int] = {}
        for row in range(self._table.rowCount()):
            key_item = self._table.item(row, 0)
            val_item = self._table.item(row, 1)
            if key_item and val_item:
                key = key_item.text().strip()
                if key:
                    try:
                        result[key] = int(val_item.text())
                    except ValueError:
                        pass
        return result if result else None

    def set_value(self, value: dict[str, int] | None) -> None:
        self._table.setRowCount(0)
        if not value:
            return
        for key, val in value.items():
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(key)))
            self._table.setItem(row, 1, QTableWidgetItem(str(val)))


class ChannelPickerDialog(QDialog):
    """Two-column dialog for selecting a subset of channels."""

    def __init__(self, available: list[str], initial_selected: list[str] | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Channels")
        self.setMinimumSize(520, 420)

        selected_set = set(initial_selected) if initial_selected else set()
        left_items = [c for c in available if c not in selected_set]
        right_items = list(initial_selected) if initial_selected else []

        layout = QVBoxLayout(self)

        content = QHBoxLayout()

        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Available"))
        self._left = QListWidget()
        self._left.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._left.addItems(left_items)
        self._left.itemDoubleClicked.connect(self._add_selected)
        left_col.addWidget(self._left)

        mid_col = QVBoxLayout()
        mid_col.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        mid_col.addStretch()
        for label, slot in (
            ("&Add >", self._add_selected),
            ("Add a&ll >>", self._add_all),
            ("&Remove <", self._remove_selected),
            ("Re&move all <<", self._remove_all),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            mid_col.addWidget(btn)
        mid_col.addStretch()

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Selected"))
        self._right = QListWidget()
        self._right.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._right.addItems(right_items)
        self._right.itemDoubleClicked.connect(self._remove_selected)
        right_col.addWidget(self._right)

        content.addLayout(left_col)
        content.addLayout(mid_col)
        content.addLayout(right_col)
        layout.addLayout(content)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_selected(self) -> None:
        for item in self._left.selectedItems():
            self._right.addItem(item.text())
            self._left.takeItem(self._left.row(item))

    def _add_all(self) -> None:
        while self._left.count():
            self._right.addItem(self._left.takeItem(0).text())

    def _remove_selected(self) -> None:
        for item in self._right.selectedItems():
            self._left.addItem(item.text())
            self._right.takeItem(self._right.row(item))

    def _remove_all(self) -> None:
        while self._right.count():
            self._left.addItem(self._right.takeItem(0).text())

    def selected_channels(self) -> list[str] | None:
        if self._right.count() == 0:
            return None
        return [self._right.item(i).text() for i in range(self._right.count())]


class ChannelPickerWidget(QWidget):
    """Button that opens a ChannelPickerDialog; get_value() returns list[str] | None."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._available: list[str] = []
        self._selected: list[str] = []
        self._is_all = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._btn = QPushButton("All &channels")
        self._btn.clicked.connect(self._open_picker)
        layout.addWidget(self._btn)
        layout.addStretch()

    def set_available_channels(self, channels: list[str]) -> None:
        self._available = list(channels)
        if not self._is_all:
            avail_set = set(self._available)
            self._selected = [c for c in self._selected if c in avail_set]
        self._update_btn()

    def _update_btn(self) -> None:
        if self._is_all:
            n = len(self._available)
            suffix = f" ({n})" if n > 0 else ""
            self._btn.setText(f"All &channels{suffix}")
        else:
            self._btn.setText(f"{len(self._selected)} &channel(s) selected")

    def _open_picker(self) -> None:
        initial = None if self._is_all else self._selected
        dlg = ChannelPickerDialog(self._available, initial, parent=self)
        if dlg.exec():
            result = dlg.selected_channels()
            if result is None:
                self._is_all = True
                self._selected = []
            else:
                self._is_all = False
                self._selected = result
            self._update_btn()

    def get_value(self) -> list[str] | None:
        return None if self._is_all else list(self._selected)

    def set_value(self, value: list[str] | None) -> None:
        if value is None:
            self._is_all = True
            self._selected = []
        else:
            self._is_all = False
            self._selected = list(value)
        self._update_btn()
