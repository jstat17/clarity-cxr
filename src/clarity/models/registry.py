"""Model registry: ``(name, size)`` → builder, resolved lazily.

The loss registry uses decorators because every loss module is imported at
package import time for registration. Models cannot work that way: importing
every model module would pull optional dependencies (``kornia``,
``torchxrayvision``, timm variants) into every run. So the catalogue stores
``"module:function"`` strings and imports only the requested model.

Usage
-----
Declare a model (no import happens here)::

    declare_model("clarity", "s-16", target="clarity.models.clarity:get_clarity_s_16")

or register an already-imported builder (useful for adapters and out-of-tree
models)::

    @register_model("densenet121-cxr14")
    def _build(num_classes, weights, **kwargs): ...

Then::

    get_model("resnet", "50", num_classes=14, weights=None)

``size=WILDCARD`` ("*") matches any requested size; a model declared with an
explicit size wins over a wildcard declaration. Builders use the uniform
signature ``builder(num_classes, weights, **kwargs)``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

#: size key that matches any requested size
WILDCARD = "*"


class ModelNotFoundError(ValueError):
    """Unknown ``(name, size)`` or an unreleased model requested as public."""

    def __init__(self, name: str, size: str, available: list[str], *, reason: str = "unknown"):
        self.name = str(name)
        self.size = str(size)
        self.available = list(available)
        listed = ", ".join(self.available) if self.available else "<none>"
        super().__init__(f"{reason} model {self.name!r}/{self.size!r}; available: {listed}")


@dataclass(frozen=True)
class ModelEntry:
    """One catalogue entry: either a dotted target or a concrete builder."""

    name: str
    size: str
    public: bool = False
    target: str | None = None
    builder: Callable[..., Any] | None = None

    @property
    def label(self) -> str:
        return self.name if self.size == WILDCARD else f"{self.name}-{self.size}"

    def __post_init__(self) -> None:
        if (self.target is None) == (self.builder is None):
            raise ValueError(f"model {self.label}: declare exactly one of target/builder")


_REGISTRY: dict[tuple[str, str], ModelEntry] = {}
_CATALOGUE_LOADED = False


def _ensure_catalogue() -> None:
    """Load the model declarations once.

    The public models are declared in the factory module, so
    ``import clarity.models.registry`` works on its own.  An import failure
    inside that module is not swallowed.
    """
    global _CATALOGUE_LOADED
    if _CATALOGUE_LOADED:
        return
    _CATALOGUE_LOADED = True  # set first: the modules import this one
    package = __package__ or "clarity.models"
    for module_name in (f"{package}.factory",):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == module_name or module_name.startswith(missing + "."):
                continue
            raise
        break


def _normalise(name: str, size: str | None) -> tuple[str, str]:
    return (str(name).casefold(), str(size or WILDCARD).casefold())


def declare_model(
    name: str,
    size: str = WILDCARD,
    *,
    target: str | None = None,
    builder: Callable[..., Any] | None = None,
    public: bool = False,
) -> ModelEntry:
    """Declare a model without importing its module."""
    key = _normalise(name, size)
    if key in _REGISTRY:
        raise ValueError(f"model {_REGISTRY[key].label!r} is already registered")
    if target is not None and ":" not in target:
        raise ValueError(
            f"model {name!r}/{size!r}: target must be 'module:function', got {target!r}"
        )
    entry = ModelEntry(
        name=str(name),
        size=str(size or WILDCARD),
        public=bool(public),
        target=target,
        builder=builder,
    )
    _REGISTRY[key] = entry
    return entry


def register_model(name: str, size: str = WILDCARD, *, public: bool = False):
    """Decorator form of :func:`declare_model` for a concrete builder."""

    def _decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
        declare_model(name, size, builder=builder, public=public)
        return builder

    return _decorator


def available_models(*, public_only: bool = False) -> list[str]:
    """Sorted labels (``name`` or ``name-size``) of the declared models."""
    _ensure_catalogue()
    labels = {entry.label for entry in _REGISTRY.values() if entry.public or not public_only}
    return sorted(labels, key=str.casefold)


def declared_models(*, public_only: bool = False) -> list[ModelEntry]:
    """Catalogue entries (declared or already resolved), sorted by label."""
    _ensure_catalogue()
    unique: dict[tuple[str, str], ModelEntry] = {}
    for entry in _REGISTRY.values():
        if entry.public or not public_only:
            # wildcard resolutions are cached under the requested size too;
            # collapse them back onto the declared entry.
            unique.setdefault((entry.name.casefold(), entry.size.casefold()), entry)
    return sorted(unique.values(), key=lambda entry: entry.label.casefold())


def _lookup(name: str, size: str) -> ModelEntry | None:
    key = _normalise(name, size)
    entry = _REGISTRY.get(key)
    if entry is not None:
        return entry
    return _REGISTRY.get((key[0], WILDCARD))


def _sizes_for(name: str) -> list[str]:
    key = str(name).casefold()
    return sorted(
        entry.size
        for (entry_name, _), entry in _REGISTRY.items()
        if entry_name == key and entry.size != WILDCARD
    )


def _not_found(name: str, size: str, *, public_only: bool) -> ModelNotFoundError:
    sizes = _sizes_for(name)
    available = available_models(public_only=public_only)
    if sizes:
        reason = "unknown size for"
        available = [f"{name}-{candidate}" for candidate in sizes]
    else:
        reason = "unknown"
    return ModelNotFoundError(name, size, available, reason=reason)


def resolve_model(name: str, size: str = WILDCARD, *, public_only: bool = False) -> ModelEntry:
    """Return the entry for ``(name, size)``, importing its module if needed."""
    _ensure_catalogue()
    entry = _lookup(name, size)
    if entry is None:
        raise _not_found(name, size, public_only=public_only)
    if public_only and not entry.public:
        raise ModelNotFoundError(name, size, available_models(public_only=True), reason="unreleased")

    if entry.builder is not None:
        return entry

    module_name, _, function_name = entry.target.partition(":")
    module = importlib.import_module(module_name)
    try:
        builder = getattr(module, function_name)
    except AttributeError as exc:  # pragma: no cover - catalogue typo
        raise RuntimeError(
            f"model {entry.label!r}: {module_name} has no attribute {function_name!r}"
        ) from exc

    resolved = ModelEntry(
        name=entry.name,
        size=entry.size,
        public=entry.public,
        builder=builder,
    )
    # Cache under the requested key and the declared key so later lookups skip
    # the import; the catalogue declaration stays for introspection.
    _REGISTRY.setdefault(_normalise(name, size), resolved)
    _REGISTRY.setdefault(_normalise(entry.name, entry.size), resolved)
    return resolved


def get_model(
    name: str, size: str = WILDCARD, *args: Any, public_only: bool = False, **kwargs: Any
) -> Any:
    """Build a model by name/size.

    ``*args``/``**kwargs`` are forwarded to the builder, whose uniform
    signature is ``builder(num_classes, weights, **kwargs)``. ``public_only``
    restricts resolution to the subset the companion ships.
    """
    entry = resolve_model(name, size, public_only=public_only)
    return entry.builder(*args, **kwargs)
