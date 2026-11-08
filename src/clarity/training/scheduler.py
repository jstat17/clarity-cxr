"""Cosine learning-rate schedule with linear warmup.

Own implementation of the schedule previously vendored from torchtune
(BSD-3); the warmup/cosine arithmetic is unchanged so existing runs replay
exactly.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineLR(LambdaLR):
    """``LambdaLR`` carrying the schedule metadata used in run logs."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
        num_warmup_steps: int = 0,
        num_training_steps: int = 1,
        num_cycles: float = 0.5,
    ):
        super().__init__(optimizer, lr_lambda, last_epoch)
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles

    def __str__(self):
        return textwrap.dedent(f"""\
                CosineLR(
                    last_epoch={self.last_epoch},
                    num_warmup_steps={self.num_warmup_steps},
                    num_training_steps={self.num_training_steps},
                    num_cycles={self.num_cycles}
                )""")

    def dict_repr(self) -> dict[str, dict]:
        """JSON-friendly description persisted in ``hyperparams.json``."""
        return {
            "CosineLR": {
                "last_epoch": repr(self.last_epoch),
                "num_warmup_steps": repr(self.num_warmup_steps),
                "num_training_steps": repr(self.num_training_steps),
                "num_cycles": repr(self.num_cycles),
            }
        }


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup followed by a cosine decay to zero.

    This matches the Hugging Face / torchtune arithmetic used for the paper's
    runs (``num_cycles=0.5`` decays from the max value to zero).
    """

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine_lr_multiple = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return max(0.0, cosine_lr_multiple)

    return CosineLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
        last_epoch=last_epoch,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
