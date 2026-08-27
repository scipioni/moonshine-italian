"""Configuration loading for config.yaml + .env.

All pipeline behavior is driven by config.yaml; private values come from
a git-ignored .env file (see .env.example).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_PATH = REPO_ROOT / ".env"

VALID_HW_PROFILES = ("rocm12g", "strix", "cuda")
VALID_TRAIN_PROFILES = ("smoke", "final")
VALID_BASES = ("streaming-small", "non_streaming_base", "lora")
VALID_QUANTIZATION = ("int8", "fp16", "none")


def load_env(env_path: Path = ENV_PATH) -> dict[str, str]:
    """Parse .env (KEY=VALUE lines) without external dependencies."""
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class ConfigError(Exception):
    """Raised when config.yaml is missing keys or has invalid values."""


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate config.yaml; returns the parsed tree."""
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    cfg = yaml.safe_load(path.read_text())
    _validate(cfg)
    return cfg


def _require(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise ConfigError(f"config.yaml: missing required section '{key}'")
    return cfg[key]


def _validate(cfg: dict[str, Any]) -> None:
    base = _require(cfg, "base_model")
    for key in ("id", "selected_base"):
        if key not in base:
            raise ConfigError(f"config.yaml: base_model.{key} is required")
    if base["selected_base"] not in VALID_BASES:
        raise ConfigError(
            f"config.yaml: base_model.selected_base must be one of {VALID_BASES}"
        )

    paths = _require(cfg, "paths")
    for key in ("data", "results", "artifacts"):
        if key not in paths:
            raise ConfigError(f"config.yaml: paths.{key} is required")

    hw = _require(cfg, "hardware_profiles")
    for name in VALID_HW_PROFILES:
        if name not in hw:
            raise ConfigError(f"config.yaml: hardware_profiles.{name} is required")
        for key in ("device", "accelerator_kind", "batch_size", "precision", "num_workers"):
            if key not in hw[name]:
                raise ConfigError(f"config.yaml: hardware_profiles.{name}.{key} is required")

    training = _require(cfg, "training")
    profiles = _require(training, "profiles")
    for name in VALID_TRAIN_PROFILES:
        if name not in profiles:
            raise ConfigError(f"config.yaml: training.profiles.{name} is required")
        prof = profiles[name]
        for key in ("output_dir", "gate"):
            if key not in prof:
                raise ConfigError(
                    f"config.yaml: training.profiles.{name}.{key} is required"
                )
        # Step budget must be either a fixed step count or epoch-derived
        # (training-pipeline spec: "Step budget is expressed independently
        # of gradient accumulation") -- exactly one of the two forms, so a
        # run's length is never ambiguous between them.
        has_fixed_steps = "max_steps" in prof
        has_epoch_budget = "target_epochs" in prof
        if has_fixed_steps == has_epoch_budget:
            raise ConfigError(
                f"config.yaml: training.profiles.{name} must set exactly one "
                "of 'max_steps' (with eval_steps/save_steps) or "
                "'target_epochs' (with eval_every_epoch_fraction)"
            )
        if has_fixed_steps:
            for key in ("eval_steps", "save_steps"):
                if key not in prof:
                    raise ConfigError(
                        f"config.yaml: training.profiles.{name}.{key} is "
                        "required alongside max_steps"
                    )
        elif "eval_every_epoch_fraction" not in prof:
            raise ConfigError(
                f"config.yaml: training.profiles.{name}.eval_every_epoch_fraction "
                "is required alongside target_epochs"
            )
        if prof["gate"] not in _require(_require(cfg, "evaluation"), "gates"):
            raise ConfigError(
                f"config.yaml: training.profiles.{name}.gate references an unknown gate"
            )

    evaluation = _require(cfg, "evaluation")
    streaming = _require(evaluation, "streaming")
    hop = streaming.get("hop_ms")
    if not isinstance(hop, (int, float)) or not 32 <= hop <= 100:
        raise ConfigError("config.yaml: evaluation.streaming.hop_ms must be within 32..100")

    export_cfg = _require(cfg, "export")
    if export_cfg.get("quantization") not in VALID_QUANTIZATION:
        raise ConfigError(
            f"config.yaml: export.quantization must be one of {VALID_QUANTIZATION}"
        )

    _require(cfg, "smoke")
    _require(cfg, "datasets")
    _require(cfg, "preparation")
    _require(cfg, "board")


@dataclass(frozen=True)
class ResolvedProfile:
    """Merged view of a hardware profile and a training profile."""

    name: str                      # e.g. "smoke"
    hardware: str                  # e.g. "rocm12g"
    device: str
    accelerator_kind: str
    batch_size: int
    precision: str
    num_workers: int
    ort_provider: str
    steps_per_second_min: float | None
    output_dir: Path
    max_steps: int | None
    eval_steps: int | None
    save_steps: int | None
    target_epochs: float | None
    eval_every_epoch_fraction: float | None
    curriculum: list[dict[str, Any]]
    gate: str
    datasets: list[str]

    def run_metadata(self) -> dict[str, Any]:
        return {
            "profile": self.name,
            "hardware": self.hardware,
            "device": self.device,
            "accelerator_kind": self.accelerator_kind,
            "batch_size": self.batch_size,
            "precision": self.precision,
            "num_workers": self.num_workers,
            "steps_per_second_min": self.steps_per_second_min,
        }


def resolve_profile(
    cfg: dict[str, Any],
    hardware: str,
    training_profile: str,
    *,
    validate_hw: bool = True,
) -> ResolvedProfile:
    """Combine a hardware profile with a training profile into one view."""
    if validate_hw and hardware not in VALID_HW_PROFILES:
        raise ConfigError(
            f"unknown hardware profile '{hardware}'. Valid profiles: "
            + ", ".join(VALID_HW_PROFILES)
        )
    if training_profile not in VALID_TRAIN_PROFILES:
        raise ConfigError(
            f"unknown training profile '{training_profile}'. Valid profiles: "
            + ", ".join(VALID_TRAIN_PROFILES)
        )
    hw = cfg["hardware_profiles"][hardware]
    tp = cfg["training"]["profiles"][training_profile]
    return ResolvedProfile(
        name=training_profile,
        hardware=hardware,
        device=hw["device"],
        accelerator_kind=hw["accelerator_kind"],
        batch_size=hw["batch_size"],
        precision=hw["precision"],
        num_workers=hw["num_workers"],
        ort_provider=hw["ort_provider"],
        steps_per_second_min=hw.get("steps_per_second_min"),
        output_dir=REPO_ROOT / tp["output_dir"],
        max_steps=tp.get("max_steps"),
        eval_steps=tp.get("eval_steps"),
        save_steps=tp.get("save_steps"),
        target_epochs=tp.get("target_epochs"),
        eval_every_epoch_fraction=tp.get("eval_every_epoch_fraction"),
        curriculum=list(tp.get("curriculum") or []),
        gate=tp["gate"],
        # Training manifests to concatenate: data/prepared/<name>/train.jsonl
        # for each name. Defaults to the original mls-only behavior for any
        # profile (e.g. smoke) that doesn't declare it -- smoke's own loading
        # path in train_loop.py ignores this and uses its fixed slice_manifest.
        datasets=list(tp.get("datasets") or ["mls"]),
    )


def absolute_path(cfg: dict[str, Any], key: str) -> Path:
    return (REPO_ROOT / cfg["paths"][key]).resolve()


def hf_token() -> str | None:
    env = {k: v for k, v in os.environ.items() if k == "HF_TOKEN"}
    token = env.get("HF_TOKEN") or load_env().get("HF_TOKEN")
    return token or None


def env_var(name: str) -> str | None:
    """Read a private value from the environment, then .env (see hf_token)."""
    return os.environ.get(name) or load_env().get(name) or None
