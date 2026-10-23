"""Shared fixtures: real-data Open Ephys integration tests and synthetic SpikeGLX sessions."""
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io
import open_ephys.analysis as oe
from open_ephys.analysis.formats.BinaryRecording import BinaryRecording
from dh5io import DH5File

from ephyscon.config import load_config_from_file
from ephyscon.convert_open_ephys_to_dh5 import convert_open_ephys_recording_to_dh5

_DATA_DIR = Path(__file__).parent / "data"
_SESSION_NAME = "Test_2026-03-12_16-32-08"

DATA_FOLDER = _DATA_DIR / _SESSION_NAME
GOLDEN_DH5_PATH = _DATA_DIR / f"{_SESSION_NAME}_exp1_rec1.dh5"
GOLDEN_CONFIG_PATH = _DATA_DIR / f"{_SESSION_NAME}_exp1_rec1.config.json"
SPIKEGLX_TEMPLATES = _DATA_DIR / "spikeglx_np1"
SPIKEGLX_CROP_SESSION = _DATA_DIR / "SOMxCD1_172_20250109_crop"

skip_if_no_data = pytest.mark.skipif(
    not (DATA_FOLDER.exists() and GOLDEN_DH5_PATH.exists()),
    reason="Real data not available",
)


@pytest.fixture(scope="session")
def oe_recording() -> BinaryRecording:
    session = oe.Session(str(DATA_FOLDER))
    return session.recordnodes[0].recordings[0]


@pytest.fixture(scope="session")
def golden_dh5() -> DH5File:
    f = DH5File(str(GOLDEN_DH5_PATH))
    yield f
    f._file.close()


@pytest.fixture(scope="session")
def fresh_dh5_path(oe_recording, tmp_path_factory):
    """Re-run the conversion with the golden config and return path to the result."""
    config = load_config_from_file(GOLDEN_CONFIG_PATH)
    tmp_dir = tmp_path_factory.mktemp("fresh")
    session_name = str(tmp_dir / _SESSION_NAME)
    convert_open_ephys_recording_to_dh5(oe_recording, session_name, config)
    exp = oe_recording.experiment_index + 1
    rec = oe_recording.recording_index + 1
    return tmp_dir / f"{_SESSION_NAME}_exp{exp}_rec{rec}.dh5"


def _write_stream(run_dir: Path, run: str, stream: str, start: str, duration_s: float) -> None:
    template = (SPIKEGLX_TEMPLATES / f"template.{stream}.meta").read_text()
    meta = dict(line.split("=", 1) for line in template.splitlines() if "=" in line)
    n_chan = int(meta["nSavedChans"])
    rate = float(meta.get("imSampRate") or meta["niSampRate"])
    n_samples = int(duration_s * rate)
    n_bytes = n_samples * n_chan * 2

    replacements = {
        "fileSizeBytes": str(n_bytes),
        "fileTimeSecs": str(n_samples / rate),
        "fileCreateTime": start,
        "fileName": f"D:/data/{run}/{run}_t0.{stream}.bin",
    }
    text = template
    for key, value in replacements.items():
        text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)

    base = run_dir / f"{run}_t0.{stream}"
    base.with_suffix(base.suffix + ".meta").write_text(text)
    np.zeros((n_samples, n_chan), dtype=np.int16).tofile(base.with_suffix(base.suffix + ".bin"))


@pytest.fixture
def make_spikeglx_session(tmp_path: Path) -> Callable[..., Path]:
    """Create a tiny SpikeGLX session: one folder per (run_name, start_time) pair."""

    def make(
        runs: Sequence[tuple[str, str]] = (("3_rfmapping_g0", "2025-01-09T13:27:45"),),
        streams: Sequence[str] = ("imec0.ap", "imec0.lf", "nidq"),
        duration_s: float = 0.05,
        name: str = "SOMxCD1_172_20250109_Recording1",
    ) -> Path:
        session = tmp_path / name
        for run, start in runs:
            run_dir = session / run
            run_dir.mkdir(parents=True)
            for stream in streams:
                _write_stream(run_dir, run, stream, start, duration_s)
        return session

    return make


def write_v73(path: Path, groups: dict[str, dict[str, np.ndarray]]) -> None:
    """Write a minimal MATLAB v7.3 file containing scalar structs of double arrays."""
    with h5py.File(path, "w", userblock_size=512) as f:
        for group_name, fields in groups.items():
            g = f.create_group(group_name)
            g.attrs["MATLAB_class"] = np.bytes_("struct")
            for name, value in fields.items():
                ds = g.create_dataset(name, data=np.atleast_2d(np.asarray(value, float)).T)
                ds.attrs["MATLAB_class"] = np.bytes_("double")
    with open(path, "r+b") as f:
        f.write(b"MATLAB 7.3 MAT-file, created by ephyscon tests".ljust(128))


def write_stimstage_paradigm(
    run_dir: Path, folder: str, params: list[dict], stimulus_order: Sequence[float]
) -> Path:
    paradigm = run_dir / folder
    paradigm.mkdir(parents=True)
    cell = np.empty((1, len(params)), dtype=object)
    for i, p in enumerate(params):
        cell[0, i] = p
    scipy.io.savemat(paradigm / "params.mat", {"params": cell})
    order = np.asarray(stimulus_order, float)
    write_v73(
        paradigm / "stimulation.mat",
        {
            "stimResults": {
                "stimulusOrder": order,
                "stimulusCount": np.array([np.sum(~np.isnan(order))]),
                "repetition": np.array([1.0]),
            },
            "screenInfo": {"FrameRate": np.array([60.0])},
        },
    )
    return paradigm


def real_session_path() -> Path | None:
    path = os.environ.get("EPHYSCON_TEST_SESSION")
    return Path(path) if path else None
