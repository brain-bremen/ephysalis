"""Settings of the pipeline steps: one model per step, the ``[steps.<step>]`` tables of ``conversion_batch.toml``
(every field required there) and the options of the step's command (``ephysalis lfp --rate-hz 1000 ...``, from the
same model: :func:`add_step_settings_arguments`). Empty strings and lists stand for "not set" (TOML has no null);
the field descriptions say what that means.

A step writes the settings it ran with to its output folder (``<work>/lfp/settings.json``, ...,
:func:`write_step_settings`); the exports describe the processing from them (:meth:`LfpSettings.filtering`, ...).
"""

from __future__ import annotations

import argparse
import json
import typing
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

STEP_SETTINGS_FILENAME = "settings.json"


# fields that choose what to process or how fast, not what comes out
NOT_IN_OUTPUT = frozenset({"probes", "n_jobs", "cache_dir", "scratch_dir", "keep_binary"})


class StepSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def output_settings(self) -> dict[str, Any]:
        """The settings that change the step's output (without :data:`NOT_IN_OUTPUT`)."""
        return self.model_dump(mode="json", exclude=set(NOT_IN_OUTPUT))


class AntiAliasResampling(StepSettings):
    """Polyphase resampling (``scipy.signal.resample_poly``) with a Kaiser-windowed FIR low-pass; the defaults are
    SciPy's design (:mod:`ephysalis.processing.resampling`)."""

    rate_hz: float = Field(default=1000.0, description="Output sampling rate (Hz)")
    max_denominator: int = Field(default=10000, description="Largest denominator of the rational up/down ratio "
                                 "approximating rate_hz / input rate")
    anti_alias_kaiser_beta: float = Field(default=5.0, description="Kaiser window beta of the anti-alias FIR")
    anti_alias_half_length: int = Field(default=10, description="Half length of the anti-alias FIR in output samples "
                                        "(10 * max(up, down) upsampled input samples); it reaches this far into the "
                                        "future and the past")
    anti_alias_cutoff: float = Field(default=1.0, description="Cutoff of the anti-alias FIR as a fraction of the "
                                     "output Nyquist frequency")

    def resampling(self) -> str:
        reach_ms = self.anti_alias_half_length / self.rate_hz * 1000
        cutoff_hz = self.anti_alias_cutoff * self.rate_hz / 2
        return (f"polyphase resampling to {self.rate_hz:g} Hz (Kaiser-windowed FIR, beta {self.anti_alias_kaiser_beta:g}, "
                f"cutoff {cutoff_hz:g} Hz, {2 * self.anti_alias_half_length + 1} output samples long), zero phase "
                f"(acausal, reaches {reach_ms:g} ms into the future)")


class AlignClocksSettings(StepSettings):
    cache_dir: str = Field(default="", description="Sync edge cache (empty: <work-dir>/sync)")


class CalibrateSettings(StepSettings):
    pass


class ExtractTrialsSettings(StepSettings):
    pass


class ExtractEventsSettings(StepSettings):
    pulse_lines: list[str] = Field(default_factory=list, description=(
        "Roles of digital lines with in_events = false whose pulses are kept anyway: in the exported files' "
        "trigger pulses (DH5 /Intervals/<role>, NWB trigger_pulses), not in the events or the BIDS events.tsv "
        "(e.g. the 100 Hz frame pulses of a stimulus PC)"))


class ExtractBehaviorSettings(StepSettings):
    analog_roles: list[str] = Field(default_factory=list, description=(
        "Roles of analog aux channels ([aux_channels.analog]) written besides the eye position, as 1 kHz bin means "
        "in V (analog/<run>.npz; DH5 CONT 3604... in this order, NWB behavior TimeSeries), e.g. photodiodes or the "
        "pupil diameter; a run without a role's channel leaves it out"))


class LfpSettings(AntiAliasResampling):
    """LFP: the LF stream (or the wideband stream) resampled with the anti-alias filter."""

    probes: list[str] = Field(default_factory=list, description="Only these probes (empty: all)")
    n_jobs: int = Field(default=4, description="Parallel jobs")

    def filtering(self) -> str:
        return self.resampling()


class QualitySettings(StepSettings):
    probes: list[str] = Field(default_factory=list, description="Only these probes (empty: all with "
                              "[probes.<id>.quality])")


class CsdSettings(StepSettings):
    probes: list[str] = Field(default_factory=list, description="Only these probes (empty: all with "
                              "[probes.<id>.csd])")
    n_jobs: int = Field(default=4, description="Parallel jobs")


class MuaSettings(AntiAliasResampling):
    """MUA envelope: the wideband signal high-pass filtered, rectified and resampled with the anti-alias filter."""

    probes: list[str] = Field(default_factory=list, description="Only these probes (empty: all)")
    highpass_hz: float = Field(default=300.0, description="High-pass cutoff (Hz)")
    highpass_order: int = Field(default=4, description="Order of the high-pass filter")
    highpass_type: Literal["butter", "bessel", "cheby1", "cheby2", "ellip"] = Field(
        default="butter", description="IIR design of the high-pass (scipy.signal.iirfilter ftype)")
    highpass_direction: Literal["forward-backward", "forward", "backward"] = Field(
        default="forward-backward", description="forward-backward: zero phase (acausal), forward: causal")
    highpass_margin_ms: float = Field(default=100.0, description="Margin read on each side of a chunk for the "
                                      "high-pass (ms)")
    rectify: bool = Field(default=True, description="Take the absolute value after the high-pass")
    n_jobs: int = Field(default=4, description="Parallel jobs")

    def filtering(self) -> str:
        from ephysalis.processing.resampling import highpass_reach_ms

        phase = "zero phase" if self.highpass_direction == "forward-backward" else self.highpass_direction
        reach = highpass_reach_ms(self) if self.highpass_direction != "forward" else 0.0
        highpass = (f"{self.highpass_order}th-order {self.highpass_type} high-pass {self.highpass_hz:g} Hz "
                    f"{self.highpass_direction} ({phase}"
                    + (f", reaches about {reach:g} ms into the future)" if reach else ")"))
        return f"{highpass}, {'rectified, ' if self.rectify else ''}{self.resampling()}"


class SpikeSortSettings(StepSettings):
    scratch_dir: str = Field(default="", description="Fast local folder for the concatenated binary (empty: the "
                             "system's temp folder)")
    probes: list[str] = Field(default_factory=list, description="Only these probes (empty: all)")
    spike_sorter: str = Field(default="", description="spikeinterface sorter, e.g. kilosort4 (empty: "
                              "[spike_sorting] of session.toml)")
    sorter_params: list[str] = Field(default_factory=list, description="Sorter parameters KEY=VALUE (values as "
                                     "JSON), on top of [spike_sorting] params")
    n_jobs: int = Field(default=4, description="Parallel jobs for copying and waveforms")
    keep_binary: bool = Field(default=False, description="Keep the local binary copy (e.g. for Phy)")


class ExportSettings(StepSettings):
    output_format: Literal["nwb", "dh5"] = Field(default="nwb", description="File format")
    bids: str = Field(default="", description="Put the file into this BIDS dataset (empty: no BIDS; BIDS-like for "
                      "DH5, BIDS only allows NWB and NIX data files)")
    output: str = Field(default="", description="Output file without bids (empty: <work-dir>/<session>.<format>)")
    existing: str = Field(default="", description="With bids: copy this existing session file instead of writing one")
    subject: str = Field(default="", description="BIDS subject label (empty: the subject id without "
                         "non-alphanumerics)")
    session_label: str = Field(default="", description="BIDS session label (empty: the session id without the "
                               "subject prefix)")
    power_line_frequency_hz: float = Field(default=50.0, description="Mains frequency (Hz)")
    dataset_name: str = Field(default="", description="Name in dataset_description.json (empty: the folder name)")
    lfp: bool = Field(default=True, description="Include the LFP")
    per_run: bool = Field(default=False, description="One file per run, on the run's own clock (BIDS: run-<n>, n = "
                          "position of the run among the included runs)")
    raw: bool = Field(default=False, description="Include the raw data (wideband streams and analog aux inputs)")
    compression: bool = Field(default=True, description="NWB: gzip-compress the LFP")


# steps in the order they run
STEP_SETTINGS: dict[str, type[StepSettings]] = {
    "align-clocks": AlignClocksSettings,
    "calibrate": CalibrateSettings,
    "extract-trials": ExtractTrialsSettings,
    "extract-events": ExtractEventsSettings,
    "extract-behavior": ExtractBehaviorSettings,
    "lfp": LfpSettings,
    "quality": QualitySettings,
    "csd": CsdSettings,
    "mua": MuaSettings,
    "spike-sort": SpikeSortSettings,
    "export": ExportSettings,
}
# extra flags of a setting on the command line
_ALIASES = {("export", "output"): ["-o"]}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def add_step_settings_arguments(parser: argparse.ArgumentParser, step: str) -> None:
    """Options of ``step``'s command from its settings model (defaults and help from the fields)."""
    for name, field in STEP_SETTINGS[step].model_fields.items():
        default = field.get_default(call_default_factory=True)
        annotation = field.annotation
        help_text = field.description or ""
        flags = [*_ALIASES.get((step, name), []), _flag(name)]
        if annotation is bool:
            parser.add_argument(*flags, dest=name, action=argparse.BooleanOptionalAction, default=default,
                                help=help_text)
        elif typing.get_origin(annotation) is list:
            parser.add_argument(*flags, dest=name, nargs="*", default=default, help=help_text)
        elif typing.get_origin(annotation) is Literal:
            parser.add_argument(*flags, dest=name, choices=typing.get_args(annotation), default=default,
                                help=f"{help_text} (default {default})")
        else:
            parser.add_argument(*flags, dest=name, type=annotation, default=default,
                                help=f"{help_text} (default {default!r})" if default != "" else help_text)


def step_settings_from_args(step: str, args: argparse.Namespace) -> Any:
    model = STEP_SETTINGS[step]
    return model.model_validate({name: getattr(args, name) for name in model.model_fields})


def step_settings_argv(step: str, settings: dict[str, Any]) -> list[str]:
    """Command line options of ``step`` for ``settings`` (all of them: the command line is explicit too)."""
    model = STEP_SETTINGS[step]
    values = model.model_validate(settings)
    argv: list[str] = []
    for name in model.model_fields:
        value = getattr(values, name)
        if isinstance(value, bool):
            argv.append(_flag(name) if value else _flag("no_" + name))
        elif isinstance(value, list):
            argv += [_flag(name), *map(str, value)]
        else:
            argv += [_flag(name), str(value)]
    return argv


def step_settings_from_options(step: str, options: dict[str, Any]) -> Any:
    """Settings of ``step`` from the options of its provenance record (``vars(args)``: also other keys)."""
    model = STEP_SETTINGS[step]
    return model.model_validate({k: v for k, v in options.items() if k in model.model_fields})


def missing_step_settings(step: str, settings: dict[str, Any]) -> list[str]:
    return [name for name in STEP_SETTINGS[step].model_fields if name not in settings]


def write_step_settings(folder: Path, settings: StepSettings) -> Path:
    """``<folder>/settings.json``: the settings a step ran with (read by the exports)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / STEP_SETTINGS_FILENAME
    path.write_text(json.dumps(settings.model_dump(mode="json"), indent=1) + "\n", encoding="utf-8")
    return path


def read_step_settings(folder: Path, model: type[StepSettings]) -> Any:
    path = folder / STEP_SETTINGS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the step again (it records its settings there)")
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
