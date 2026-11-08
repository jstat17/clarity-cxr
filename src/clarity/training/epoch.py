"""One training epoch (forward/backward/accumulation) for the training loop."""

from __future__ import annotations

import logging

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from clarity.utils.runtime import get_autocast_context

logger = logging.getLogger("clarity.training.loop")


def train_one_epoch(
    epoch: int,
    num_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    accumulation_factor: int = 1,
    use_amp: bool = False,
) -> float:
    """Train the model for one epoch.

    Args:
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs.
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for the training set.
        optimizer (optim.Optimizer): The optimizer to be used.
        criterion (nn.Module): The loss function to be used.
        device (str): The device to be used ('cuda' or 'cpu').
        accumulation_factor (int, optional): The gradient accumulation factor. Defaults to 1.
        use_amp (bool, optional): Use bfloat16 autocast for the model forward
            pass and loss computation. Defaults to False.

    Returns:
        _type_: _description_
    """
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

    # initialize the gradient accumulation counter
    accumulation_counter = 0
    n_batches = len(train_loader)

    for i, batch in enumerate(pbar):
        # get model input and send to device
        X = batch[0]
        X = X.to(device, non_blocking=True)

        # get model targets and send to device
        Y = batch[1:]
        Y = [y.to(device, non_blocking=True) for y in Y]
        if len(Y) == 1:
            Y = Y[0]

        # zero all gradients
        if accumulation_counter == 0:
            optimizer.zero_grad(set_to_none=True)

        # Only the model forward pass and loss computation use autocast.
        # Keep backward and optimizer.step() outside autocast.
        with get_autocast_context(device, enabled=use_amp):
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)

        # backward pass
        loss.backward()

        # Increment the accumulation counter
        accumulation_counter += 1

        # If the accumulation factor is reached or it's the last batch, perform optimization
        if accumulation_counter == accumulation_factor or i == n_batches - 1:
            # Guard against a single rare non-finite-gradient event (e.g. an
            # AMP underflow) killing a multi-day run: skip this update and
            # warn loudly instead of poisoning the weights.
            #
            # Single-sync finite check: a grad tensor contains a non-finite
            # element iff its L2 norm is non-finite, so fusing the per-parameter
            # isfinite reductions into one _foreach_norm + a single all-reduce
            # check replaces one device sync per parameter (~330 round-trips
            # per optimizer step) with exactly one.
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if len(grads) < len(list(model.parameters())):
                # rare path: some grads are None (e.g. unused parameters);
                # fall back to the per-parameter check to stay exact
                grads_finite = all(
                    p.grad is None or bool(th.isfinite(p.grad).all()) for p in model.parameters()
                )
            else:
                norms = th._foreach_norm(grads)
                grads_finite = bool(th.stack(norms).isfinite().all())
            if grads_finite:
                optimizer.step()
            else:
                logger.warning(
                    "non-finite gradients at epoch %s batch %s; skipping "
                    "optimizer step", epoch + 1, i,
                )
                optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

        # update running loss
        running_loss += loss.item()
        pbar.set_postfix(loss=running_loss / (i + 1))

    return running_loss / n_batches
