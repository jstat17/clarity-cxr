"""Shared localization threshold constants.

:mod:`localization.onebox` can import them without an import cycle (the evaluator
modules import ``localization.onebox``; ``localization.onebox`` must not import the
evaluator modules).

The values are frozen standard constants and must stay bit-identical to the
historical definitions they replaced:

* ``THRESHOLDS_HEATMAP``: the 91-level absolute heatmap threshold grid
  ``0.05 .. 0.95`` (step 0.01) used by the legacy SingleCut scan.
* ``THRESHOLDS_SCORE``: the five val score thresholds ``[0.1, 0.25, 0.5,
  0.75, 0.9]`` used by the val5 calibration criterion.

``THRESHOLDS_SCORE`` intentionally stays a ``list`` (not an array) because
dozens of call sites use it as an iterable default argument.
"""

from __future__ import annotations

import numpy as np

THRESHOLDS_HEATMAP = np.linspace(5, 95, 91) / 100
THRESHOLDS_SCORE = [0.1, 0.25, 0.5, 0.75, 0.9]
