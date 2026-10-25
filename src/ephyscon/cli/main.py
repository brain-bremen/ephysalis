import argparse
from ephyscon import convert_open_ephys_session
from ephyscon.config import load_config_from_file
from ephyscon.inspect import format_session_info, inspect_session, validate_session_path
from ephyscon.cli.session_commands import add_session_commands
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import sys
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

if sys.platform == "win32":
    import winreg
import logging

logger = logging.getLogger(__name__)


def pick_open_ephys_session_via_dialog() -> Path | None:
    # Set default initialdir to ~/Documents/Open Ephys if it exists
    default_dir = Path.home() / "Documents" / "Open Ephys"

    if sys.platform == "win32":

        def get_last_dir():
            try:
                with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER, r"Software\IfH", 0, winreg.KEY_READ
                ) as key:
                    value, _ = winreg.QueryValueEx(key, "oe_to_dh_last_dir")
                    if value and Path(value).exists():
                        return value
            except Exception:
                return None

        def set_last_dir(path):
            try:
                with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\IfH") as key:
                    winreg.SetValueEx(
                        key, "oe_to_dh_last_dir", 0, winreg.REG_SZ, str(path)
                    )
            except Exception:
                pass

        last_dir = get_last_dir()
        initialdir = last_dir if last_dir else None
    else:
        last_dir_file = Path.home() / ".oe_to_dh_last_dir"
        initialdir = None
        if last_dir_file.exists():
            try:
                with open(last_dir_file, "r") as f:
                    last_dir = f.read().strip()
                    if last_dir and Path(last_dir).exists():
                        initialdir = last_dir
            except Exception:
                pass

        def set_last_dir(path):
            try:
                with open(last_dir_file, "w") as f:
                    f.write(str(path))
            except Exception:
                pass

    initialdir = initialdir or default_dir

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(
        title="Select Open Ephys session folder", initialdir=initialdir
    )
    if path == "":
        return None

    set_last_dir(path)
    return Path(path)


def add_convert_command(sub: argparse._SubParsersAction) -> None:
    # ephyscon convert --output-folder <output_folder> --config <config.json> --tdr <path_to_tdr_file> <oe-session>
    parser = sub.add_parser(
        "convert",
        help="Convert Open Ephys recordings to DH5 format.",
        description="Convert Open Ephys recordings to DH5 format.",
    )

    parser.add_argument(
        "oe_session", type=str, nargs="*", help="Path(s) to Open Ephys session folder(s)."
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder for DH5 files. Defaults to each session's parent folder.",
    )

    parser.add_argument(
        "--config", type=str, help="Path to the configuration JSON file."
    )
    parser.add_argument("--tdr", type=str, help="Path to the TDR file.")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print a summary of the session contents and exit.",
    )
    parser.set_defaults(func=convert)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ephyscon",
        description="Preprocessing of extracellular electrophysiology recordings.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    add_convert_command(sub)
    add_session_commands(sub)

    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)  # for the processing record
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    return args.func(args) or 0


def convert(args: argparse.Namespace) -> int:
    # If no sessions provided, open a file dialog to pick one
    if not args.oe_session:
        picked = pick_open_ephys_session_via_dialog()
        if picked is None:
            return 0
        args.oe_session = [str(picked)]

    # Validate all paths up front
    session_paths: list[Path] = []
    for s in args.oe_session:
        p = Path(s)
        validate_session_path(p)
        session_paths.append(p)

    # attempt to load config from path if present
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = None

    if args.inspect:
        for session_path in session_paths:
            print(format_session_info(inspect_session(session_path)))
            print()
        return

    output_folder = Path(args.output_folder) if args.output_folder else None

    session_bar = tqdm(session_paths, desc="Sessions", unit="session", disable=len(session_paths) == 1)
    step_bar: tqdm | None = None

    def on_progress(step_name: str, done: int, total: int) -> None:
        nonlocal step_bar
        if step_bar is None or step_bar.total != total:
            if step_bar is not None:
                step_bar.close()
            step_bar = tqdm(total=total, desc=step_name, unit="unit", leave=False)
        step_bar.set_description(step_name)
        step_bar.n = done
        step_bar.refresh()

    with logging_redirect_tqdm():
        for session_path in session_bar:
            convert_open_ephys_session(session_path, output_folder=output_folder, config=config, on_progress=on_progress)
            if step_bar is not None:
                step_bar.close()
                step_bar = None

    session_bar.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        logger.error(f"Current config value: {locals().get('config', None)}")
