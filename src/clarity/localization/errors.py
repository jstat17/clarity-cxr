"""Worker-side error types shared by the evaluators."""

from __future__ import annotations


class DetectionWorkerError(RuntimeError):
    """A localization detection/reduction worker failed.

    The evaluators wrap worker exceptions in this type so the failing
    (variant, filename, abnormality) context survives to the caller.
    """
