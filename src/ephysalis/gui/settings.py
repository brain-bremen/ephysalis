import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QListView,
    QTreeView,
    QWidget,
)


def _q() -> QSettings:
    return QSettings("brain-bremen", "OEcon")


def get_last_session_dir() -> str | None:
    v = _q().value("last_session_dir")
    return str(v) if v and Path(str(v)).is_dir() else None


def set_last_session_dir(path: Path) -> None:
    _q().setValue("last_session_dir", str(path.parent))


def get_last_config_path() -> str | None:
    v = _q().value("last_config_path")
    return str(v) if v and Path(str(v)).is_file() else None


def set_last_config_path(path: str) -> None:
    _q().setValue("last_config_path", path)


def _pick_session_dirs_windows(
    parent: QWidget, title: str, initial: str
) -> "list[Path] | None":
    """
    Use the native Windows IFileOpenDialog COM interface to pick multiple folders.
    Returns a list of paths, or None if the call fails (so the caller can fall back).
    Supports UNC network shares and the full Windows shell namespace because we
    deliberately omit FOS_FORCEFILESYSTEM (which Qt adds and which blocks network paths).
    """
    import logging

    log = logging.getLogger(__name__)

    try:
        import comtypes
        import comtypes.client
    except Exception as e:
        log.warning("comtypes import failed, falling back to Qt dialog: %s", e)
        return None

    try:
        from ctypes import c_ulong, c_wchar_p
        from ctypes.wintypes import HWND

        # IFileOpenDialog option flags
        FOS_PICKFOLDERS = 0x00000020
        FOS_ALLOWMULTISELECT = 0x00000200
        FOS_FILEMUSTEXIST = 0x00001000

        # SIGDN value for retrieving a Win32 filesystem path from an IShellItem
        SIGDN_FILESYSPATH = 0x80058000

        # HRESULT for user cancellation: HRESULT_FROM_WIN32(ERROR_CANCELLED)
        HRESULT_CANCELLED = -2147023673  # 0x800704C7 as signed 32-bit

        # ------------------------------------------------------------------
        # COM interface definitions
        # Each COMMETHOD param is a tuple: (["in"/"out"], ctype, "name")
        # ------------------------------------------------------------------

        class IShellItem(comtypes.IUnknown):
            _iid_ = comtypes.GUID("{43826D1E-E718-42EE-BC55-A1E261C37BFE}")
            _methods_ = [
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "BindToHandler",
                    (["in"], comtypes.POINTER(comtypes.IUnknown), "pbc"),
                    (["in"], comtypes.POINTER(comtypes.GUID), "bhid"),
                    (["in"], comtypes.POINTER(comtypes.GUID), "riid"),
                    (["out"], comtypes.POINTER(comtypes.c_void_p), "ppv"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetParent",
                    (
                        ["out"],
                        comtypes.POINTER(comtypes.POINTER(comtypes.IUnknown)),
                        "ppsi",
                    ),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetDisplayName",
                    (["in"], c_ulong, "sigdnName"),
                    (["out"], comtypes.POINTER(c_wchar_p), "ppszName"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetAttributes",
                    (["in"], c_ulong, "sfgaoMask"),
                    (["out"], comtypes.POINTER(c_ulong), "psfgaoAttribs"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "Compare",
                    (["in"], comtypes.POINTER(comtypes.IUnknown), "psi"),
                    (["in"], c_ulong, "hint"),
                    (["out"], comtypes.POINTER(c_ulong), "piOrder"),
                ),
            ]

        class IShellItemArray(comtypes.IUnknown):
            _iid_ = comtypes.GUID("{B63EA76D-1F85-456F-A19C-48159EFA858B}")
            _methods_ = [
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "BindToHandler",
                    (["in"], comtypes.POINTER(comtypes.IUnknown), "pbc"),
                    (["in"], comtypes.POINTER(comtypes.GUID), "bhid"),
                    (["in"], comtypes.POINTER(comtypes.GUID), "riid"),
                    (["out"], comtypes.POINTER(comtypes.c_void_p), "ppv"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetPropertyStore",
                    (["in"], c_ulong, "flags"),
                    (["in"], comtypes.POINTER(comtypes.GUID), "riid"),
                    (["out"], comtypes.POINTER(comtypes.c_void_p), "ppv"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetPropertyDescriptionList",
                    (["in"], comtypes.POINTER(comtypes.GUID), "keyType"),
                    (["in"], comtypes.POINTER(comtypes.GUID), "riid"),
                    (["out"], comtypes.POINTER(comtypes.c_void_p), "ppv"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetAttributes",
                    (["in"], c_ulong, "AttribFlags"),
                    (["in"], c_ulong, "sfgaoMask"),
                    (["out"], comtypes.POINTER(c_ulong), "psfgaoAttribs"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetCount",
                    (["out"], comtypes.POINTER(c_ulong), "pdwNumItems"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetItemAt",
                    (["in"], c_ulong, "dwIndex"),
                    (["out"], comtypes.POINTER(comtypes.POINTER(IShellItem)), "ppsi"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "EnumItems",
                    (["out"], comtypes.POINTER(comtypes.c_void_p), "ppenumShellItems"),
                ),
            ]

        class IFileDialog(comtypes.IUnknown):
            _iid_ = comtypes.GUID("{42F85136-DB7E-439C-85F1-E4075D135FC8}")
            _methods_ = [
                # Slot 3: inherited from IModalWindow — must come first
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "Show",
                    (["in"], HWND, "hwndOwner"),
                ),
                # Slots 4–26: IFileDialog-specific methods in SDK vtable order
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetFileTypes",
                    (["in"], c_ulong, "cFileTypes"),
                    (["in"], comtypes.c_void_p, "rgFilterSpec"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetFileTypeIndex",
                    (["in"], c_ulong, "iFileType"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetFileTypeIndex",
                    (["out"], comtypes.POINTER(c_ulong), "piFileType"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "Advise",
                    (["in"], comtypes.c_void_p, "pfde"),
                    (["out"], comtypes.POINTER(c_ulong), "pdwCookie"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "Unadvise",
                    (["in"], c_ulong, "dwCookie"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetOptions",
                    (["in"], c_ulong, "fos"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetOptions",
                    (["out"], comtypes.POINTER(c_ulong), "pfos"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetDefaultFolder",
                    (["in"], comtypes.POINTER(IShellItem), "psi"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetFolder",
                    (["in"], comtypes.POINTER(IShellItem), "psi"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetFolder",
                    (["out"], comtypes.POINTER(comtypes.POINTER(IShellItem)), "ppsi"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetCurrentSelection",
                    (["out"], comtypes.POINTER(comtypes.POINTER(IShellItem)), "ppsi"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetFileName",
                    (["in"], c_wchar_p, "pszName"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetFileName",
                    (["out"], comtypes.POINTER(c_wchar_p), "pszName"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetTitle",
                    (["in"], c_wchar_p, "pszTitle"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetOkButtonLabel",
                    (["in"], c_wchar_p, "pszText"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetFileNameLabel",
                    (["in"], c_wchar_p, "pszLabel"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetResult",
                    (["out"], comtypes.POINTER(comtypes.POINTER(IShellItem)), "ppsi"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "AddPlace",
                    (["in"], comtypes.POINTER(IShellItem), "psi"),
                    (["in"], c_ulong, "fdap"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetDefaultExtension",
                    (["in"], c_wchar_p, "pszDefaultExtension"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "Close",
                    (["in"], comtypes.HRESULT, "hr"),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetClientGuid",
                    (["in"], comtypes.POINTER(comtypes.GUID), "guid"),
                ),
                comtypes.COMMETHOD([], comtypes.HRESULT, "ClearClientData"),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "SetFilter",
                    (["in"], comtypes.c_void_p, "pFilter"),
                ),
            ]

        class IFileOpenDialog(IFileDialog):
            _iid_ = comtypes.GUID("{D57C7288-D4AD-4768-BE02-9D969532D960}")
            _methods_ = [
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetResults",
                    (
                        ["out"],
                        comtypes.POINTER(comtypes.POINTER(IShellItemArray)),
                        "ppenum",
                    ),
                ),
                comtypes.COMMETHOD(
                    [],
                    comtypes.HRESULT,
                    "GetSelectedItems",
                    (
                        ["out"],
                        comtypes.POINTER(comtypes.POINTER(IShellItemArray)),
                        "ppsai",
                    ),
                ),
            ]

        # ------------------------------------------------------------------
        # Create and configure the dialog
        # ------------------------------------------------------------------

        CLSID_FileOpenDialog = comtypes.GUID("{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}")
        ifd: IFileOpenDialog = comtypes.client.CreateObject(  # type: ignore[assignment]
            CLSID_FileOpenDialog, interface=IFileOpenDialog
        )

        # Read existing options and add our flags.
        # Deliberately omit FOS_FORCEFILESYSTEM so that network shares (UNC paths)
        # and other non-filesystem shell namespace items remain accessible.
        existing_options = ifd.GetOptions()
        ifd.SetOptions(
            existing_options
            | FOS_PICKFOLDERS
            | FOS_ALLOWMULTISELECT
            | FOS_FILEMUSTEXIST
        )
        ifd.SetTitle(title)

        hwnd = int(parent.winId()) if parent else 0

        # Show() raises a COMError with HRESULT_CANCELLED when the user dismisses
        # the dialog — that is normal; any other exception is a real error.
        try:
            ifd.Show(hwnd)
        except comtypes.COMError as e:
            if e.hresult == HRESULT_CANCELLED:
                return []
            raise

        # ------------------------------------------------------------------
        # Collect selected paths from the IShellItemArray
        # ------------------------------------------------------------------
        items: IShellItemArray = ifd.GetResults()  # type: ignore[assignment]
        count = items.GetCount()

        paths: list[Path] = []
        for i in range(count):
            item: IShellItem = items.GetItemAt(i)  # type: ignore[assignment]
            try:
                name = item.GetDisplayName(SIGDN_FILESYSPATH)
                if name:
                    paths.append(Path(name))
            except comtypes.COMError:
                # Item has no filesystem path (e.g. a virtual shell folder) — skip it.
                pass

        return paths

    except Exception as e:
        log.warning(
            "IFileOpenDialog failed, falling back to Qt dialog: %s", e, exc_info=True
        )
        return None


def _pick_session_dirs_qt_fallback(
    parent: QWidget, title: str, initial: str
) -> list[Path]:
    """Non-native Qt folder picker with multi-select. Used on Linux/Mac."""
    dialog = QFileDialog(parent, title, initial)
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

    for view in (dialog.findChild(QListView, "listView"), dialog.findChild(QTreeView)):
        if view:
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    if not dialog.exec():
        return []
    return [Path(p) for p in dialog.selectedFiles()]


def pick_session_dirs(
    parent: QWidget,
    title: str = "Select Open Ephys session folder(s)",
    initial: str = "",
) -> list[Path]:
    """
    Open a folder-picker dialog supporting multiple directory selection.

    On Windows, uses the native IFileOpenDialog COM interface so that network
    shares (UNC paths) and the full Windows shell namespace are accessible.
    Falls back to a non-native Qt dialog on Linux/Mac (or if COM initialisation fails).
    """
    if sys.platform == "win32":
        result = _pick_session_dirs_windows(parent, title, initial)
        if result is not None:
            return result
        # Fall through to Qt fallback if COM approach failed for any reason

    return _pick_session_dirs_qt_fallback(parent, title, initial)
