"""Per-session configuration (``session.toml``).

The session config replaces the legacy Excel sheets (font colour = run included),
``Depths.txt``, ``BrokenChannelsProbe<N>_0Index.txt`` and ``folders2Combine.txt``. It is
created once per session (``ephyscon init``), can be edited by hand, and is versioned
alongside the outputs for provenance.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, Field

from ephyscon.session import Session

SCHEMA_VERSION = 1
SESSION_CONFIG_FILENAME = "session.toml"


class RunConfig(BaseModel):
    name: str
    include: bool = True
    stimulus_system: str | None = Field(
        default=None, description="Stimulus system plugin; auto-detected if not set"
    )
    notes: str = ""


class SortingConfig(BaseModel):
    """Spike sorter and its parameters (any sorter of ``spikeinterface.sorters``)."""

    sorter: str = Field(
        default="kilosort4", description="spikeinterface sorter name, e.g. kilosort4, mountainsort5, spykingcircus2"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Sorter parameters that differ from the sorter's (and ephyscon's) defaults"
    )


class ProbeGeometry(BaseModel):
    """Probe geometry from the probeinterface library or a probeinterface JSON file."""

    manufacturer: str | None = Field(default=None, description="probeinterface library, e.g. cambridgeneurotech")
    model: str | None = Field(default=None, description="probeinterface library, e.g. ASSY-37-E-1")
    file: str | None = Field(default=None, description="probeinterface JSON file (relative to session.toml)")
    wiring: str | None = Field(default=None, description="probeinterface wiring to the headstage, e.g. ASSY-156>RHD2164")


class ProbeConfig(BaseModel):
    stream: str | None = Field(
        default=None, description="Input stream of a probe the reader does not detect (non-Neuropixels), "
        "e.g. 'Acquisition_Board-100.Rhythm Data'"
    )
    channels: list[str] | None = Field(default=None, description="Channel ids of the probe in `stream` (default: all)")
    geometry: ProbeGeometry | None = None
    area: str | None = Field(default=None, description="Target brain area, e.g. V1")
    insertion_depth_um: float | None = Field(
        default=None, description="Depth of the probe tip below the brain surface"
    )
    shank_depth_um: float | None = Field(
        default=None, description="'ShankDepth' from the lab Excel sheets"
    )
    bad_channels: list[int] = Field(
        default_factory=list, description="0-based channel indices excluded from sorting"
    )
    serial_number: str | None = None
    sorting: SortingConfig | None = Field(
        default=None, description="Sorter for this probe, instead of the session's [sorting]"
    )
    notes: str = ""


class SubjectConfig(BaseModel):
    """Subject metadata for the NWB file (NWB ``Subject``; DANDI requires species and sex)."""

    species: str | None = Field(default=None, description="Latin binomial, e.g. Mus musculus")
    sex: str | None = Field(default=None, description="M, F, U or O")
    age: str | None = Field(default=None, description="ISO 8601 duration, e.g. P90D")
    date_of_birth: str | None = Field(default=None, description="ISO 8601 date")
    strain: str | None = None
    genotype: str | None = None
    description: str = ""


class AuxChannelMap(BaseModel):
    """Roles of the auxiliary (NIDQ) channels.

    ``analog`` maps a role to the analog input index (XA<n> / AI<n>); ``digital`` maps a role
    to a bit of the digital word. Defaults reproduce the MATLAB pipeline
    (``multiParadigmArtefactRemovalAndSpikesortingComb_contEye_CR.m``) for 8-channel rigs.
    """

    analog: dict[str, int] = Field(
        default_factory=lambda: {"laser_red": 0, "photodiode": 2, "laser_green": 4, "laser_blue": 6}
    )
    digital: dict[str, int] = Field(
        default_factory=lambda: {"sync": 1, "photodiode_invalid": 2, "paradigm": 3, "wheel": 7}
    )

    @classmethod
    def vstim_openephys_default(cls) -> AuxChannelMap:
        """Open Ephys rigs with VStim: eye position on AI0/AI2, VStim trigger line 1 (bit 0)
        marks trial starts, line 8 (bit 7) the screen frames."""
        return cls(analog={"eye_x": 0, "eye_y": 2}, digital={"trial_start": 0, "frame": 7})

    @classmethod
    def legacy_default(cls, n_analog: int) -> AuxChannelMap:
        if n_analog == 4:
            # older 4-channel rigs: AI0..AI3 = red laser, photodiode, green laser, blue laser
            return cls(analog={"laser_red": 0, "photodiode": 1, "laser_green": 2, "laser_blue": 3})
        return cls()


class SessionConfig(BaseModel):
    schema_version: int = SCHEMA_VERSION
    session_id: str
    subject_id: str
    input_format: str | None = None
    experimenter: list[str] = Field(default_factory=list)
    lab: str | None = None
    institution: str | None = None
    description: str = ""
    subject: SubjectConfig = Field(default_factory=SubjectConfig)
    runs: list[RunConfig] = Field(default_factory=list)
    probes: dict[str, ProbeConfig] = Field(default_factory=dict)
    sorting: SortingConfig = Field(default_factory=SortingConfig)
    aux_channels: AuxChannelMap = Field(default_factory=AuxChannelMap)
    notes: str = ""

    def run_config(self, name: str) -> RunConfig | None:
        return next((r for r in self.runs if r.name == name), None)

    @property
    def included_runs(self) -> list[str]:
        return [r.name for r in self.runs if r.include]


def save_session_config(config: SessionConfig, path: str | Path) -> None:
    data = config.model_dump(mode="json", exclude_none=True)
    Path(path).write_text(tomli_w.dumps(data), encoding="utf-8")


def load_session_config(path: str | Path) -> SessionConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("schema_version", SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema version {version}, newer than supported ({SCHEMA_VERSION})"
        )
    return SessionConfig.model_validate(data)


_SESSION_NAME_RE = re.compile(r"^(?P<subject>.+?)_(?P<date>\d{8})(?:_|$)")


def default_session_config(session: Session) -> SessionConfig:
    """Config with all runs included and probe metadata taken from the recording files."""
    match = _SESSION_NAME_RE.match(session.name)
    subject = match["subject"] if match else session.name

    probes: dict[str, ProbeConfig] = {}
    for run in session.runs:
        for pid, info in run.probes.items():
            probes.setdefault(pid, ProbeConfig(serial_number=info.serial_number))

    n_analog = next((r.extra["aux"]["n_analog"] for r in session.runs if r.extra.get("aux")), 8)
    aux_channels = (AuxChannelMap.vstim_openephys_default() if session.format.name == "openephys"
                    else AuxChannelMap.legacy_default(n_analog))
    return SessionConfig(
        session_id=session.name,
        subject_id=subject,
        input_format=session.format.name,
        runs=[RunConfig(name=run.name) for run in session.runs],
        probes=probes,
        aux_channels=aux_channels,
    )
