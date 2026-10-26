"""Per-session configuration (``session.toml``).

The session config replaces the legacy Excel sheets (font colour = run included),
``Depths.txt``, ``BrokenChannelsProbe<N>_0Index.txt`` and ``folders2Combine.txt``. It is
created once per session (``ephyscon init``) and can be edited by hand. The outputs embed
the config they were made with (:mod:`ephyscon.provenance`).

Schema versions (``schema_version``, see :mod:`ephyscon.schema`):

1. first version
2. ``lab``, ``institution``, ``[subject]``, ``[sorting]``; probes: ``stream``, ``channels``,
   ``geometry``, ``sorting`` (non-Neuropixels probes, per-probe sorter)
3. ``[events]`` (TTL lines left out of the events, offset of the network event codes)
4. ``[calibration]`` (gains from a calibration-sine recording), ``[line_noise]`` (mains removal
   with the recorded mains reference), probes: ``csd`` (current source density)
5. ``[events.lines]`` (active level and description of the trigger lines), ``[[events.markers]]``
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, Field

from ephyscon.schema import Migration, migrate
from ephyscon.session import Session

SCHEMA_VERSION = 5
SESSION_CONFIG_FILENAME = "session.toml"
# MIGRATIONS[n]: raw data of schema version n -> n + 1
MIGRATIONS: dict[int, Migration] = {
    1: lambda data: data,  # 2 only added fields, all with defaults
    2: lambda data: data,  # 3 added [events], with defaults
    3: lambda data: data,  # 4 added [calibration], [line_noise] and probes.csd, all off by default
    4: lambda data: data,  # 5 added [events.lines] and [[events.markers]], empty by default
}


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


class CsdConfig(BaseModel):
    """Current source density of a 2-D electrode array (Borisov's ``dh_csd`` V3.1 [RBF], :mod:`ephyscon.csd`):
    Gaussian RBF interpolation of the LFP through the input electrodes, CSD = -Laplacian /
    specific resistance at the output electrodes. Defaults: the lab's parameters for the Fritz array."""

    grid_spacing_m: float = Field(default=0.003, description="Array grid spacing (m); the RBF width is "
                                  "grid_spacing_m x rbf_parameter")
    rbf_parameter: float = Field(default=2.2758, description="RBF interpolation parameter (dh_csd 'rbfparam')")
    specific_resistance_ohm_m: float = Field(default=1.0, description="Specific resistance of the tissue (Ohm m)")
    exclude_inputs: list[str] = Field(
        default_factory=list, description="Channel ids not used as inputs (e.g. a bad electrode); channels "
        "without calibration are left out as well. Outputs: all channels of the probe")


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
    csd: CsdConfig | None = Field(default=None, description="Compute the CSD of this probe (ephyscon csd); "
                                  "needs the probe geometry")
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


class TriggerLineConfig(BaseModel):
    """A digital line with a role in ``[aux_channels.digital]`` (:mod:`ephyscon.annotations`)."""

    active: Literal["high", "low"] = Field(default="high", description="Level of the line during a pulse, "
                                           "e.g. 'low' for a lever line that is low while pressed")
    description: str = Field(default="", description="Meaning of the pulses (DH5 /Intervals, BIDS events.json)")


class MarkerConfig(BaseModel):
    """A time marker from the pulses of a trigger line or from the trials (:mod:`ephyscon.annotations`)."""

    name: str = Field(description="Marker name, e.g. lever_press, derived_sample_onset")
    role: str = Field(description="Role of the line ([aux_channels.digital]), or 'trial' for the trials")
    edge: Literal["onset", "offset"] = Field(default="onset", description="Start or end of the pulses (trials)")
    offset_ms: float = Field(default=0.0, description="Added to the time; a marker with an offset is derived, "
                             "not measured")
    description: str = ""


class EventsConfig(BaseModel):
    """Events of the runs (``events/<run>.csv``, DH5 ``EV02``, NWB ``events`` table) and the pulses
    and markers made from them (DH5 ``/Intervals``, ``/Markers``; :mod:`ephyscon.annotations`)."""

    exclude_roles: list[str] = Field(
        default_factory=lambda: ["sync", "wheel"],
        description="Digital line roles ([aux_channels]) left out of the events, e.g. fast pulse trains",
    )
    network_events_offset: int = Field(
        default=1000, description="Added to the VStim network event words for the DH5 event codes "
        "(TTL lines are +line / -line)"
    )
    lines: dict[str, TriggerLineConfig] = Field(
        default_factory=dict, description="Active level and description of the lines, by role "
        "(default: active high, described by the role)"
    )
    markers: list[MarkerConfig] = Field(default_factory=list, description="Markers besides trial_start")


class CalibrationConfig(BaseModel):
    """Channel gains from a recording of a calibration sine on all inputs (``ephyscon calibrate``,
    :mod:`ephyscon.calibration`). Without ``run`` and ``file`` the gains of the recording are used."""

    run: str | None = Field(default=None, description="Calibration run of this session, e.g. 15hz_after_2005_03_30")
    file: str | None = Field(default=None, description="calibration.json of another session's work dir (relative "
                             "to session.toml), for sessions without their own calibration recording")
    amplitude_uv: float = Field(default=100.0, description="Amplitude of the calibration sine at the electrodes (µV)")
    frequency_range_hz: tuple[float, float] = Field(default=(14.5, 16.5), description="Search range of the sine "
                                                    "frequency (the lab's generator ran at about 15.4 Hz)")
    min_relative_amplitude: float = Field(default=0.3, description="Channels whose sine is below this fraction of "
                                          "the median amplitude of the probe have no calibration signal")
    missing: str = Field(default="nan", description="Gain of channels without calibration signal: 'nan' (left "
                         "out of the CSD) or 'median' (median gain of the probe)")


class LineNoiseConfig(BaseModel):
    """Mains removal in the LFP (``ephyscon lfp``) with the recorded mains reference: port of the lab's
    ``dh_remove_50hz`` (M. Borisov 2004, :mod:`ephyscon.linenoise`). Needs a 1 kHz LFP."""

    enabled: bool = False
    reference_role: str = Field(default="mains", description="Analog role ([aux_channels.analog]) of the "
                                "recorded mains")
    min_cycle_fraction: float = Field(default=0.95, description="The reference of a run is used only if this "
                                      "fraction of its cycles lasts 19-21 ms ...")
    min_peak_db: float = Field(default=20.0, description="... and its 50 Hz peak is this far above the spectrum "
                               "5-20 Hz away (else the run is left unchanged)")
    subtract_reference_mean: bool = Field(default=False, description="Find the zero crossings of the reference "
                                          "after subtracting its mean (as the Fritz/Mini BIDS conversions) instead "
                                          "of on the reference as recorded (dh_remove_50hz)")


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
    events: EventsConfig = Field(default_factory=EventsConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    line_noise: LineNoiseConfig = Field(default_factory=LineNoiseConfig)
    notes: str = ""

    def run_config(self, name: str) -> RunConfig | None:
        return next((r for r in self.runs if r.name == name), None)

    @property
    def included_runs(self) -> list[str]:
        return [r.name for r in self.runs if r.include]


def save_session_config(config: SessionConfig, path: str | Path) -> None:
    data = config.model_dump(mode="json", exclude_none=True)
    Path(path).write_text(tomli_w.dumps(data), encoding="utf-8")


def session_config_from_dict(data: dict[str, Any], source: object = "session config") -> SessionConfig:
    """Config from raw data of any supported schema version (older ones are migrated)."""
    return SessionConfig.model_validate(migrate(data, SCHEMA_VERSION, MIGRATIONS, source=source))


def load_session_config(path: str | Path) -> SessionConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return session_config_from_dict(data, source=path)


_SESSION_NAME_RE = re.compile(r"^(?P<subject>.+?)_(?P<date>\d{8})(?:_|$)")


def default_session_config(session: Session) -> SessionConfig:
    """Config with all runs included and probe metadata taken from the recording files."""
    match = _SESSION_NAME_RE.match(session.name)
    subject = match["subject"] if match else session.name

    probes: dict[str, ProbeConfig] = {}
    for run in session.runs:
        for pid, info in run.probes.items():
            probes.setdefault(pid, ProbeConfig(serial_number=info.serial_number))

    with_aux = [r for r in session.runs if r.extra.get("aux")] or session.runs
    aux_channels = session.format.default_aux_channels(with_aux[0] if with_aux else None)
    return SessionConfig(
        session_id=session.name,
        subject_id=subject,
        input_format=session.format.name,
        runs=[RunConfig(name=run.name) for run in session.runs],
        probes=probes,
        aux_channels=aux_channels,
    )
