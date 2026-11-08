"""Small array- and model-type helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class LogitMask:
    """Container for the classification logits a model should expose.

    Accepted forms:

    * ``None`` / ``slice(None)`` — all logits (the default);
    * ``slice`` — passed through, e.g. ``slice(2, 5)``;
    * ``str`` — comma-separated inclusive ranges/indices, e.g. ``"0-1, 4"``;
    * a two-int ``tuple`` — historical half-open range, e.g. ``(0, 2)`` -> 0, 1;
    * any other iterable of ints — explicit indices.

    The mask is applied to the class dimension of logits and localization
    maps, so a checkpoint with many output classes can emit only the classes
    a downstream dataset actually supervises.
    """

    indices: slice | list[int]

    def __init__(self, mask: Iterable[int] | str | slice | None = None) -> None:
        if mask is None or (isinstance(mask, slice) and mask == slice(None)):
            self.indices = slice(None)
            return

        if isinstance(mask, slice):
            self.indices = mask
            return

        if isinstance(mask, str):
            indices: list[int] = []
            for part in mask.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start, end = (int(value) for value in part.split("-", 1))
                    if end < start:
                        raise ValueError(f"invalid descending logit range {part!r}")
                    indices.extend(range(start, end + 1))
                else:
                    indices.append(int(part))
            self.indices = _validated(indices)
            return

        # Historical form: a (start, end) pair means a half-open range. A list
        # of two ints, by contrast, is the explicit index list users expect.
        if (
            isinstance(mask, tuple)
            and len(mask) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in mask)
        ):
            self.indices = _validated(list(range(mask[0], mask[1])))
            return

        try:
            indices = list(mask)
        except TypeError as exc:
            raise TypeError(f"unsupported logit mask {mask!r}") from exc
        self.indices = _validated([int(index) for index in indices])

    def __iter__(self):
        if isinstance(self.indices, slice):
            raise TypeError("an unbounded LogitMask has no finite iteration")
        return iter(self.indices)

    def __len__(self) -> int:
        if isinstance(self.indices, slice):
            raise TypeError("an unbounded LogitMask has no length")
        return len(self.indices)

    def __str__(self) -> str:
        return str(self.indices)

    def __repr__(self) -> str:
        return f"LogitMask({self.indices!r})"


def _validated(indices: list[int]) -> list[int]:
    if any(index < 0 for index in indices):
        raise ValueError(f"logit indices must be non-negative: {indices}")
    return indices


def contiguity_order(arr: np.ndarray) -> str | None:
    """Return the memory-contiguity order used when flattening an array.

    ``"C"`` wins when an array is both C- and F-contiguous (the historical
    behaviour); ``None`` means neither layout applies.
    """
    if arr.flags["C_CONTIGUOUS"]:
        return "C"
    if arr.flags["F_CONTIGUOUS"]:
        return "F"
    return None
