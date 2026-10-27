"""``ephysalis`` command line: the pipeline steps of :mod:`ephysalis.cli.session_commands`."""

import argparse
import logging
import sys

from ephysalis.cli.session_commands import add_session_commands

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ephysalis",
        description="Preprocessing of extracellular electrophysiology recordings (SpikeGLX, Open Ephys) "
        "into NWB or DH5 files, optionally in a BIDS dataset.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    add_session_commands(sub)

    args = parser.parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.argv = list(sys.argv[1:] if argv is None else argv)  # for the processing record
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
