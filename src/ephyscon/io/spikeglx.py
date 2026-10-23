"""SpikeGLX reader.

Expected layout (one folder per gate/run, as written by SpikeGLX)::

    <session>/
        3_rfmappingFine_g0/
            3_rfmappingFine_g0_t0.imec0.ap.bin/.meta
            3_rfmappingFine_g0_t0.imec0.lf.bin/.meta   (Neuropixels 1.0 only)
            3_rfmappingFine_g0_t0.imec1.ap.bin/.meta
            3_rfmappingFine_g0_t0.nidq.bin/.meta
            1_stimulusStageRF/                          (stimulus-system specific)
        5_DoseResponseRB_g0/
        ...

Probe-per-folder layouts (``<run>_g0/<run>_g0_imec0/...``) are found as well because
meta files are searched one level deep.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ephyscon.io.base import InputFormat
from ephyscon.session import ProbeInfo, Run, Session, StreamKind
from ephyscon.sync import Edges, digital_edges

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

    from ephyscon.session_config import AuxChannelMap

logger = logging.getLogger(__name__)

# <run>_g<gate>_t<trigger>.<stream>.meta, stream = imec<N>.ap | imec<N>.lf | nidq | obx<N>
_META_RE = re.compile(r"^(?P<run>.+_g\d+)_t(?P<trigger>\d+)\.(?P<stream>imec\d+\.(?:ap|lf)|nidq|obx\d+)\.meta$")
# the imec SY channel carries the sync input on bit 6
IMEC_SYNC_BIT = 6


def read_meta(path: Path) -> dict[str, str]:
    """Parse a SpikeGLX ``.meta`` file into a dict (leading ``~`` of tag names is stripped)."""
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.lstrip("~")] = value
    return meta


def _num_shanks(meta: dict[str, str]) -> int:
    # snsGeomMap header: (part_number, n_shanks, shank_spacing_um, shank_width_um)
    geom = meta.get("snsGeomMap")
    if geom:
        header = geom[1 : geom.index(")")].split(",")
        if len(header) >= 2 and header[1].strip().isdigit():
            return int(header[1])
    # older files: snsShankMap header: (n_shanks, n_cols, n_rows)
    shank_map = meta.get("snsShankMap")
    if shank_map:
        header = shank_map[1 : shank_map.index(")")].split(",")
        if header[0].strip().isdigit():
            return int(header[0])
    return 1


def _num_neural_channels(meta: dict[str, str]) -> int:
    # snsApLfSy = n_ap, n_lf, n_sync
    ap, lf, _ = (int(x) for x in meta.get("snsApLfSy", "0,0,0").split(","))
    return max(ap, lf)


class SpikeGLXFormat(InputFormat):
    name: ClassVar[str] = "spikeglx"

    def can_read(self, session_path: Path) -> bool:
        return self._is_run(session_path) or any(
            self._is_run(d) for d in session_path.iterdir() if d.is_dir()
        )

    @staticmethod
    def _run_metas(run_path: Path) -> list[Path]:
        metas = [p for p in run_path.glob("*.meta") if _META_RE.match(p.name)]
        # probe-per-folder layout: <run>_g0/<run>_g0_imec0/<run>_g0_t0.imec0.ap.meta
        metas += [
            p
            for p in run_path.glob(f"{run_path.name}_imec*/*.meta")
            if _META_RE.match(p.name)
        ]
        return sorted(metas)

    @classmethod
    def _is_run(cls, path: Path) -> bool:
        return bool(cls._run_metas(path))

    def open_session(self, session_path: Path) -> Session:
        candidates = [d for d in sorted(session_path.iterdir()) if d.is_dir()]
        if self._is_run(session_path):
            # the path itself is a single run folder
            candidates = [session_path]

        runs: list[Run] = []
        for run_path in candidates:
            run = self._open_run(run_path)
            if run is not None:
                runs.append(run)

        runs.sort(key=lambda r: (r.start_time or datetime.max, r.name))
        for i, run in enumerate(runs):
            run.index = i
        return Session(name=session_path.name, path=session_path, format=self, runs=runs)

    def _open_run(self, run_path: Path) -> Run | None:
        metas = self._run_metas(run_path)
        if not metas:
            return None

        triggers = sorted({_META_RE.match(m.name)["trigger"] for m in metas})  # type: ignore[index]
        if triggers != ["0"]:
            logger.warning(
                "Run %s has triggers %s; only t0 is used for now", run_path.name, triggers
            )
        metas = [m for m in metas if _META_RE.match(m.name)["trigger"] == "0"]  # type: ignore[index]

        probes: dict[str, dict[StreamKind, str]] = {}
        probe_meta: dict[str, dict[str, str]] = {}
        aux_stream = None
        aux_info: dict[str, int] = {}
        start_time = None
        duration = None
        run_name = _META_RE.match(metas[0].name)["run"]  # type: ignore[index]
        for meta_path in metas:
            stream = _META_RE.match(meta_path.name)["stream"]  # type: ignore[index]
            meta = read_meta(meta_path)
            if start_time is None and "fileCreateTime" in meta:
                start_time = datetime.fromisoformat(meta["fileCreateTime"])
            if stream == "nidq":
                aux_stream = "nidq"
                duration = float(meta.get("fileTimeSecs", "nan"))
                # snsMnMaXaDw = multiplexed neural, multiplexed aux, direct analog, digital words
                mn, ma, xa, dw = (int(x) for x in meta.get("snsMnMaXaDw", "0,0,0,0").split(","))
                aux_info = {"n_analog": mn + ma + xa, "n_digital_words": dw}
                if "syncNiChan" in meta and meta.get("syncNiChanType") == "0":
                    aux_info["sync_bit"] = int(meta["syncNiChan"])
            elif stream.startswith("imec"):
                probe_id, band = stream.split(".")
                kinds = probes.setdefault(probe_id, {})
                kinds[StreamKind.AP if band == "ap" else StreamKind.LF] = stream
                if band == "ap":
                    kinds[StreamKind.SYNC] = f"{stream}-SYNC"
                    probe_meta[probe_id] = meta
                    duration = duration or float(meta.get("fileTimeSecs", "nan"))
            # obx (OneBox) streams are ignored for now

        probe_infos = {}
        for probe_id, kinds in sorted(probes.items()):
            meta = probe_meta.get(probe_id, {})
            probe_infos[probe_id] = ProbeInfo(
                probe_id=probe_id,
                part_number=meta.get("imDatPrb_pn"),
                serial_number=meta.get("imDatPrb_sn"),
                num_shanks=_num_shanks(meta),
                num_channels=_num_neural_channels(meta),
                has_lf=StreamKind.LF in kinds,
                streams=kinds,
            )

        return Run(
            name=run_name,
            path=run_path,
            index=-1,
            format=self,
            start_time=start_time,
            duration_s=duration,
            probes=probe_infos,
            aux_stream=aux_stream,
            extra={"meta_files": [str(m) for m in metas], "aux": aux_info},
        )

    def get_recording(
        self, run: Run, kind: StreamKind, probe_id: str | None = None
    ) -> BaseRecording:
        if kind == StreamKind.AUX:
            if run.aux_stream is None:
                raise KeyError(f"Run {run.name} has no auxiliary (NIDQ) stream")
            stream_id = run.aux_stream
        else:
            if probe_id is None:
                raise ValueError(f"probe_id is required for {kind} streams")
            try:
                stream_id = run.probes[probe_id].streams[kind]
            except KeyError:
                raise KeyError(f"Run {run.name} has no {kind} stream for {probe_id}") from None
        return _read_spikeglx(str(run.path), stream_id)

    def sync_edges(self, run: Run, probe_id: str | None, aux_channels: AuxChannelMap | None) -> Edges:
        if probe_id is not None:
            return digital_edges(self.get_recording(run, StreamKind.SYNC, probe_id), 0, IMEC_SYNC_BIT)
        bit = run.extra["aux"].get("sync_bit")
        if aux_channels is not None and "sync" in aux_channels.digital:
            if bit is not None and bit != aux_channels.digital["sync"]:
                logger.warning(
                    "Run %s: NIDQ sync bit %d from the config differs from syncNiChan=%d in the meta file",
                    run.name, aux_channels.digital["sync"], bit,
                )
            bit = aux_channels.digital["sync"]
        if bit is None:
            raise ValueError(f"Run {run.name}: unknown NIDQ sync bit")
        return digital_edges(self.get_recording(run, StreamKind.AUX), self._digital_channel(run), bit)

    @staticmethod
    def _digital_channel(run: Run) -> int:
        # the digital words follow the analog channels in the NIDQ stream
        if run.extra["aux"].get("n_digital_words", 0) < 1:
            raise KeyError(f"Run {run.name}: NIDQ stream has no digital word")
        return run.extra["aux"]["n_analog"]

    def read_aux_analog(self, run: Run, index: int) -> np.ndarray:
        if not 0 <= index < run.extra["aux"]["n_analog"]:
            raise IndexError(f"Run {run.name}: no NIDQ analog channel {index}")
        recording = self.get_recording(run, StreamKind.AUX)
        return recording.get_traces(channel_ids=[recording.channel_ids[index]])[:, 0]

    def aux_analog_gain_v(self, run: Run, index: int) -> float:
        # spikeinterface gains are in uV and include the NI range and channel gain
        return float(self.get_recording(run, StreamKind.AUX).get_channel_gains()[index]) * 1e-6

    def read_aux_digital(self, run: Run, bit: int) -> np.ndarray:
        recording = self.get_recording(run, StreamKind.AUX)
        word = recording.get_traces(channel_ids=[recording.channel_ids[self._digital_channel(run)]])[:, 0]
        return ((word.astype(np.int64) >> bit) & 1).astype(bool)


@cache
def _read_spikeglx(folder: str, stream_id: str) -> BaseRecording:
    import spikeinterface.extractors as se

    return se.read_spikeglx(folder, stream_id=stream_id)
