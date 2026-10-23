import types
import typing
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QLineEdit, QLabel, QFrame,
)

from ephyscon.gui.widgets import ChannelPickerWidget, DictEditor, ListEditor


# ---------------------------------------------------------------------------
# Type introspection helpers
# ---------------------------------------------------------------------------

def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional). Handles `T | None` and `Optional[T]`."""
    origin = typing.get_origin(annotation)
    if origin is types.UnionType or origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _is_str_enum(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, StrEnum)


def _is_list_str(t: Any) -> bool:
    return typing.get_origin(t) is list and typing.get_args(t) == (str,)


def _is_dict_str_int(t: Any) -> bool:
    return typing.get_origin(t) is dict and typing.get_args(t) == (str, int)


# ---------------------------------------------------------------------------
# Optional scalar widgets
# ---------------------------------------------------------------------------

class _OptionalSpinBox(QWidget):
    """QSpinBox with a checkbox; unchecked → None."""

    def __init__(self, default: int | None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cb = QCheckBox()
        self._spin = QSpinBox()
        self._spin.setRange(-(2**30), 2**30)
        if default is not None:
            self._cb.setChecked(True)
            self._spin.setValue(default)
        else:
            self._cb.setChecked(False)
            self._spin.setEnabled(False)
        self._cb.toggled.connect(self._spin.setEnabled)
        layout.addWidget(self._cb)
        layout.addWidget(self._spin)
        layout.addStretch()

    def get_value(self) -> int | None:
        return self._spin.value() if self._cb.isChecked() else None

    def set_value(self, v: int | None) -> None:
        if v is None:
            self._cb.setChecked(False)
        else:
            self._cb.setChecked(True)
            self._spin.setValue(v)


class _OptionalDoubleSpinBox(QWidget):
    """QDoubleSpinBox with a checkbox; unchecked → None."""

    def __init__(self, default: float | None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cb = QCheckBox()
        self._spin = QDoubleSpinBox()
        self._spin.setRange(-1e12, 1e12)
        self._spin.setDecimals(4)
        if default is not None:
            self._cb.setChecked(True)
            self._spin.setValue(float(default))
        else:
            self._cb.setChecked(False)
            self._spin.setEnabled(False)
        self._cb.toggled.connect(self._spin.setEnabled)
        layout.addWidget(self._cb)
        layout.addWidget(self._spin)
        layout.addStretch()

    def get_value(self) -> float | None:
        return self._spin.value() if self._cb.isChecked() else None

    def set_value(self, v: float | None) -> None:
        if v is None:
            self._cb.setChecked(False)
        else:
            self._cb.setChecked(True)
            self._spin.setValue(float(v))


# ---------------------------------------------------------------------------
# Widget factory
# ---------------------------------------------------------------------------

def _make_field_widget(annotation: Any, default: Any) -> QWidget:
    """Create the appropriate Qt widget for a Pydantic field annotation."""
    inner, is_optional = _unwrap_optional(annotation)

    if inner is bool:
        w = QCheckBox()
        w.setChecked(bool(default) if default is not None else False)
        return w

    if inner is int:
        if is_optional:
            return _OptionalSpinBox(default if isinstance(default, int) else None)
        w = QSpinBox()
        w.setRange(-(2**30), 2**30)
        if isinstance(default, int):
            w.setValue(default)
        return w

    if inner is float:
        if is_optional:
            return _OptionalDoubleSpinBox(float(default) if isinstance(default, (int, float)) else None)
        w = QDoubleSpinBox()
        w.setRange(-1e12, 1e12)
        w.setDecimals(4)
        if isinstance(default, (int, float)):
            w.setValue(float(default))
        return w

    if _is_str_enum(inner):
        w = QComboBox()
        for member in inner:
            w.addItem(member.value, userData=member)
        if default is not None:
            for i in range(w.count()):
                if w.itemData(i) == default:
                    w.setCurrentIndex(i)
                    break
        return w

    if inner is str:
        w = QLineEdit()
        if default is not None:
            w.setText(str(default))
        return w

    if _is_list_str(inner):
        w = ListEditor()
        if default is not None:
            w.set_value(default)
        return w

    if _is_dict_str_int(inner):
        w = DictEditor()
        if default is not None:
            w.set_value(default)
        return w

    label = QLabel("(not configurable in GUI)")
    label.setEnabled(False)
    return label


def _get_widget_value(widget: QWidget) -> Any:
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QDoubleSpinBox):
        return widget.value()
    if isinstance(widget, QComboBox):
        return widget.currentData()
    if isinstance(widget, QLineEdit):
        text = widget.text()
        return text if text else None
    if isinstance(widget, (_OptionalSpinBox, _OptionalDoubleSpinBox, ListEditor, DictEditor, ChannelPickerWidget)):
        return widget.get_value()
    return None


def _set_widget_value(widget: QWidget, value: Any) -> None:
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QSpinBox):
        if isinstance(value, int):
            widget.setValue(value)
    elif isinstance(widget, QDoubleSpinBox):
        if isinstance(value, (int, float)):
            widget.setValue(float(value))
    elif isinstance(widget, QComboBox):
        for i in range(widget.count()):
            if widget.itemData(i) == value:
                widget.setCurrentIndex(i)
                break
    elif isinstance(widget, QLineEdit):
        widget.setText(str(value) if value is not None else "")
    elif isinstance(widget, (_OptionalSpinBox, _OptionalDoubleSpinBox, ListEditor, DictEditor, ChannelPickerWidget)):
        widget.set_value(value)


def _get_field_default(model_class: type[BaseModel], field_name: str) -> Any:
    try:
        return getattr(model_class(), field_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ConfigStepWidget
# ---------------------------------------------------------------------------

class ConfigStepWidget(QWidget):
    """Auto-generates a form from any Pydantic BaseModel subclass."""

    def __init__(
        self,
        model_class: type[BaseModel],
        enabled_by_default: bool = True,
        excluded_fields: set[str] | None = None,
        field_overrides: dict[str, QWidget] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._model_class = model_class
        self._field_widgets: dict[str, QWidget] = {}
        _excluded = excluded_fields or set()
        _overrides = field_overrides or {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enable_cb = QCheckBox("&Enable this step")
        self._enable_cb.setChecked(enabled_by_default)
        layout.addWidget(self._enable_cb)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        self._form_widget = QWidget()
        form = QFormLayout(self._form_widget)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        visible_fields = {
            name: info
            for name, info in model_class.model_fields.items()
            if name not in _excluded
        }

        if not visible_fields:
            form.addRow(QLabel("No configuration options for this step."))
        else:
            for field_name, field_info in visible_fields.items():
                if field_name in _overrides:
                    widget = _overrides[field_name]
                else:
                    default = _get_field_default(model_class, field_name)
                    widget = _make_field_widget(field_info.annotation, default)
                label = (field_info.title or field_name.replace("_", " ").capitalize()) + ":"
                if field_info.description:
                    widget.setToolTip(field_info.description)
                form.addRow(label, widget)
                self._field_widgets[field_name] = widget

        layout.addWidget(self._form_widget)
        layout.addStretch()

        self._form_widget.setEnabled(enabled_by_default)
        self._enable_cb.toggled.connect(self._form_widget.setEnabled)

    def get_model(self) -> BaseModel | None:
        if not self._enable_cb.isChecked():
            return None
        values = {}
        for name, widget in self._field_widgets.items():
            # Skip non-editable fields (disabled labels) - let Pydantic use defaults
            if isinstance(widget, QLabel):
                continue
            values[name] = _get_widget_value(widget)
        return self._model_class(**values)

    def set_model(self, model: BaseModel | None) -> None:
        if model is None:
            self._enable_cb.setChecked(False)
            return
        self._enable_cb.setChecked(True)
        for field_name, widget in self._field_widgets.items():
            _set_widget_value(widget, getattr(model, field_name, None))
