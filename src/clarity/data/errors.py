"""Dataset errors."""
from __future__ import annotations


class InvalidDatasetError(Exception):
    """Raised when a DataLoader uses an invalid Dataset type."""
