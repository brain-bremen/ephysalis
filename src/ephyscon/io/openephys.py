"""Open Ephys (binary format) reader.

An Open Ephys session folder contains ``Record Node <id>/experiment<N>/recording<M>``.
Each (experiment, recording) pair becomes one :class:`~ephyscon.session.Run`. Neuropixels
streams from the Neuropix-PXI / OneBox plugins (``...ProbeA-AP``, ``...ProbeA-LFP`` or
``...ProbeA`` for 2.0 probes) are mapped to probes; all other continuous streams
(NI-DAQmx, acquisition-board ADCs) are treated as auxiliary streams.

Event (TTL / message) handling lives in the sync/stimulus layers and still comes from
oecon (to be ported).
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ephyscon.io.base import InputFormat
from ephyscon.session import ProbeInfo, Run, Session, StreamKind

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

_PROBE_STREAM_RE = re.compile(r"(?P<probe>Probe[A-Z]+)(?:-(?P<band>AP|LFP))?$")


class OpenEphysFormat(InputFormat):
    name: ClassVar[str] = "openephys"

    def can_read(self, session_path: Path) -> bool:
        return any(session_path.glob("Record Node *")) or session_path.name.startswith(
            "Record Node"
        )

    def open_session(self, session_path: Path) -> Session:
        import neo

        rawio = neo.rawio.OpenEphysBinaryRawIO(str(session_path))
        rawio.parse_header()
        stream_names = list(rawio.header["signal_streams"]["name"])

        # Only use the first record node; further nodes typically hold copies of the same data.
        record_nodes = sorted({name.split("#")[0] for name in stream_names if "#" in name})
        if record_nodes:
            stream_names = [n for n in stream_names if n.startswith(record_nodes[0] + "#")]

        probes: dict[str, dict[StreamKind, str]] = defaultdict(dict)
        aux_streams = []
        for full_name in stream_names:
            short = full_name.split("#", 1)[-1]
            match = _PROBE_STREAM_RE.search(short)
            if match is None:
                aux_streams.append(full_name)
                continue
            band = match["band"]
            probes[match["probe"]][StreamKind.LF if band == "LFP" else StreamKind.AP] = full_name

        runs: list[Run] = []
        for block in range(rawio.block_count()):
            for segment in range(rawio.segment_count(block)):
                t_start = rawio.segment_t_start(block, segment)
                t_stop = rawio.segment_t_stop(block, segment)
                probe_infos = {
                    pid: ProbeInfo(
                        probe_id=pid, has_lf=StreamKind.LF in kinds, streams=dict(kinds)
                    )
                    for pid, kinds in sorted(probes.items())
                }
                runs.append(
                    Run(
                        name=f"experiment{block + 1}_recording{segment + 1}",
                        path=session_path,
                        index=len(runs),
                        format=self,
                        duration_s=t_stop - t_start,
                        probes=probe_infos,
                        aux_stream=aux_streams[0] if aux_streams else None,
                        extra={
                            "block_index": block,
                            "segment_index": segment,
                            "aux_streams": aux_streams,
                        },
                    )
                )
        return Session(name=session_path.name, path=session_path, format=self, runs=runs)

    def get_recording(
        self, run: Run, kind: StreamKind, probe_id: str | None = None
    ) -> BaseRecording:
        if kind == StreamKind.AUX:
            if run.aux_stream is None:
                raise KeyError(f"Run {run.name} has no auxiliary stream")
            stream_name = run.aux_stream
        else:
            if probe_id is None:
                raise ValueError(f"probe_id is required for {kind} streams")
            try:
                stream_name = run.probes[probe_id].streams[kind]
            except KeyError:
                raise KeyError(f"Run {run.name} has no {kind} stream for {probe_id}") from None
        recording = _read_openephys(str(run.path), stream_name, run.extra["block_index"])
        return recording.select_segments([run.extra["segment_index"]])


@cache
def _read_openephys(folder: str, stream_name: str, block_index: int) -> BaseRecording:
    import spikeinterface.extractors as se

    return se.read_openephys(folder, stream_name=stream_name, block_index=block_index)
