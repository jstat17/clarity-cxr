"""Typed, validated YAML configuration for training and evaluation runs.

The dataclasses are the single description of a run: ``configs/*.yaml`` is
loaded into them, CLI overrides are applied with dotted keys, and the resolved
mapping is what a run folder records.  Unknown keys fail loudly rather than
being silently ignored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

import yaml

T = TypeVar("T")

from clarity.utils.paths import resolve_data_root, resolve_models_root


def _prune(value: Any) -> Any:
    """Drop ``None``/empty containers so round-trips match authored YAML."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            item = _prune(item)
            if item is None or item == {} or item == [] or item == ():
                continue
            out[key] = item
        return out
    if isinstance(value, (list, tuple)):
        return [_prune(item) for item in value]
    return value


def _checked(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {where} config key(s): {unknown}")


def _coerce(current: Any, raw: str) -> Any:
    """Coerce a CLI override string to the type of the current value."""
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean value, got {raw!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, (list, tuple)):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if current is None:
        lowered = raw.strip().lower()
        if lowered in ("none", "null", ""):
            return None
        if lowered in ("true", "false"):
            return lowered == "true"
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return raw


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and constructor options."""

    name: str = "clarity"
    size: str = "s-16"
    num_classes: int = 14
    weights: str | None = None
    init: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelConfig":
        _checked(payload, {"name", "size", "num_classes", "weights", "init", "kwargs"}, "model")
        return cls(**dict(payload))


@dataclass(frozen=True)
class DataConfig:
    """Dataset, image size and the explicit path roots."""

    image_size: int = 224
    dataset: str = "chestxray14"
    data_root: str | None = None
    images: str | None = None
    segmentations: str | None = None
    split: str | None = None
    train_localization: bool | None = None
    validate_localization: bool | None = None
    border_padding: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataConfig":
        _checked(
            payload,
            {
                "image_size",
                "dataset",
                "data_root",
                "images",
                "segmentations",
                "split",
                "train_localization",
                "validate_localization",
                "border_padding",
            },
            "data",
        )
        return cls(**dict(payload))

    def resolve_root(self) -> Path:
        """Resolve the dataset root with explicit > env precedence."""
        return resolve_data_root(self.data_root)


@dataclass(frozen=True)
class LossConfig:
    """Loss name and paper weight symbols (alpha_C, alpha_P, alpha_f, beta, gamma)."""

    name: str = "combined"
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LossConfig":
        _checked(payload, {"name", "kwargs"}, "loss")
        return cls(**dict(payload))


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer, schedule and batch options."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 0
    accumulation_factor: int = 1
    optimizer: str = "adamw"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimConfig":
        _checked(
            payload,
            {
                "learning_rate",
                "weight_decay",
                "batch_size",
                "num_epochs",
                "warmup_steps",
                "accumulation_factor",
                "optimizer",
            },
            "optim",
        )
        return cls(**dict(payload))


@dataclass(frozen=True)
class SweepConfig:
    """Attention-layer sweep selection options."""

    attention_layer_counts: tuple[int, ...] = ()
    primary: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attention_layer_counts",
            tuple(int(count) for count in self.attention_layer_counts),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SweepConfig":
        _checked(payload, {"attention_layer_counts", "primary"}, "sweep")
        return cls(**dict(payload))


@dataclass(frozen=True)
class LocalizationConfig:
    """Localization method/selection options (OneBox is the default)."""

    bbox_method: str = "onebox"
    selection: str = "joint_argmax"
    optimize_metrics_separately: bool = False
    fit_workers: int | None = None
    method: str | None = None
    attention_type: str | None = None
    only_correctly_classified: bool | None = None

    def __post_init__(self) -> None:
        if self.bbox_method not in ("onebox", "singlecut"):
            raise ValueError(
                f"unknown bbox_method {self.bbox_method!r}; expected " "'onebox' or 'singlecut'"
            )
        if self.selection not in ("joint_argmax", "mean_argmax"):
            raise ValueError(
                f"unknown selection {self.selection!r}; expected " "'joint_argmax' or 'mean_argmax'"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocalizationConfig":
        _checked(
            payload,
            {
                "bbox_method",
                "selection",
                "optimize_metrics_separately",
                "fit_workers",
                "method",
                "attention_type",
                "only_correctly_classified",
            },
            "localization",
        )
        return cls(**dict(payload))


def _apply_overrides(config: T, overrides: Sequence[str]) -> T:
    """Apply ``section.field=value`` overrides; CLI values win over YAML."""
    config = config
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not in key=value form")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        if len(parts) == 1:
            field_name = parts[0]
            if not hasattr(config, field_name):
                raise KeyError(f"unknown config field {field_name!r}")
            current = getattr(config, field_name)
            if is_dataclass(current):
                raise ValueError(f"override {item!r} targets a section; use section.field=value")
            config = replace(config, **{field_name: _coerce(current, raw)})
            continue
        if len(parts) != 2:
            raise ValueError(f"override {item!r} must use section.field=value")
        section, field_name = parts
        if not hasattr(config, section):
            raise KeyError(f"unknown config section {section!r}")
        section_obj = getattr(config, section)
        if not hasattr(section_obj, field_name):
            raise KeyError(f"unknown config field {section}.{field_name}")
        current = getattr(section_obj, field_name)
        updated = replace(section_obj, **{field_name: _coerce(current, raw)})
        config = replace(config, **{section: updated})
    return config


@dataclass(frozen=True)
class TrainConfig:
    """Complete training-run description (YAML + CLI overrides)."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    seed: int = 0
    run_folder: str | None = None
    augmentation: bool | None = None
    augmentation_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainConfig":
        payload = dict(payload)
        _checked(
            payload,
            {
                "model",
                "data",
                "loss",
                "optim",
                "sweep",
                "localization",
                "seed",
                "run_folder",
                "augmentation",
                "augmentation_kwargs",
            },
            "train",
        )
        sections = {
            "model": ModelConfig,
            "data": DataConfig,
            "loss": LossConfig,
            "optim": OptimConfig,
            "sweep": SweepConfig,
            "localization": LocalizationConfig,
        }
        kwargs = {}
        for name, factory in sections.items():
            if name in payload:
                kwargs[name] = factory.from_dict(payload[name] or {})
        for scalar in ("seed", "run_folder", "augmentation", "augmentation_kwargs"):
            if scalar in payload:
                kwargs[scalar] = payload[scalar]
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def to_dict(self) -> dict[str, Any]:
        return _prune(asdict(self))

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    @property
    def image_size(self) -> int:
        return self.data.image_size

    def with_overrides(self, overrides: Sequence[str]) -> "TrainConfig":
        """Apply ``section.field=value`` overrides; CLI values win over YAML."""
        return _apply_overrides(self, overrides)


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation-run description (classification or localization)."""

    split: str = "classify"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    models_root: str | None = None

    def __post_init__(self) -> None:
        if self.split not in ("classify", "localize"):
            raise ValueError(f"unknown split {self.split!r}; expected 'classify' or " "'localize'")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvalConfig":
        payload = dict(payload)
        _checked(payload, {"split", "model", "data", "localization", "models_root"}, "eval")
        kwargs: dict[str, Any] = {}
        for name, factory in (
            ("model", ModelConfig),
            ("data", DataConfig),
            ("localization", LocalizationConfig),
        ):
            if name in payload:
                kwargs[name] = factory.from_dict(payload[name] or {})
        for scalar in ("split", "models_root"):
            if scalar in payload:
                kwargs[scalar] = payload[scalar]
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def to_dict(self) -> dict[str, Any]:
        return _prune(asdict(self))

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    @property
    def image_size(self) -> int:
        return self.data.image_size

    def resolve_models_root(self) -> Path:
        return resolve_models_root(self.models_root)

    def with_overrides(self, overrides: Sequence[str]) -> "EvalConfig":
        """Apply ``section.field=value`` overrides; CLI values win over YAML."""
        return _apply_overrides(self, overrides)
