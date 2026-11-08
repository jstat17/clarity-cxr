"""Attention-layer sweep: specification, selection state and run artifacts.



* :mod:`.spec` — the immutable sweep specification, variant keys, result and
  manifest schemas, capability helpers.
* :mod:`.selection` — the deterministic score/tie-breaking, the per-epoch
  selection state, and manifest-based layer-count resolution.
* :mod:`.manager` — snapshot files, ``best-selection.json`` and the
  ``model_best.pth`` symlink.

The public names are re-exported here so callers keep a single import.
"""

from __future__ import annotations

from .spec import *  # noqa: F401,F403
from .selection import *  # noqa: F401,F403
from .manager import *  # noqa: F401,F403

from . import manager as _manager
from . import selection as _selection
from . import spec as _spec

for _module in (_spec, _selection, _manager):
    for _name in dir(_module):
        if _name.startswith("_") and not _name.startswith("__"):
            globals()[_name] = getattr(_module, _name)
del _module, _name
