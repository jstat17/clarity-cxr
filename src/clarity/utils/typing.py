"""Shared array-like type aliases (used by evaluation helpers)."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

Number: TypeAlias = int | float
Array1DLike: TypeAlias = np.ndarray | pd.Series | spmatrix | list
Matrix2DLike: TypeAlias = np.ndarray | pd.DataFrame | spmatrix
