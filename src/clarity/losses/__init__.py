"""Public loss subset: exactly the losses used in the CLARiTy paper.

The paper's combined objective is registered as ``combined`` and is built
from the weighted binary cross-entropy (``wbce``), background activation
suppression (``bas``) and soft Dice (``dice``) losses. ``mlsm`` is included
for the paper's classification-loss ablation.
"""

from __future__ import annotations

import clarity.losses.bas  # noqa: F401
import clarity.losses.combined  # noqa: F401
import clarity.losses.dice  # noqa: F401
import clarity.losses.mlsm  # noqa: F401
import clarity.losses.wbce  # noqa: F401
