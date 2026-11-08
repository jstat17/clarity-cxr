"""Checkpoint, performance-log and run-folder artifacts."""

from __future__ import annotations

import lzma
import pickle as pk
from pathlib import Path
from typing import Any

import natsort
import torch as th
import torch.nn as nn
import torch.optim as optim

from clarity.utils.io import convert_defaultdict_to_dict


def save_checkpoint(
    epoch: int, model: nn.Module, optimizer: optim.Optimizer, checkpoint_path: Path
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    th.save(state, checkpoint_path)


def save_performance_log(performance_dict: dict[str, Any | list[Any]], log_path: Path) -> None:
    performance_dict = convert_defaultdict_to_dict(performance_dict)
    with lzma.open(log_path, "wb") as f:
        pk.dump(performance_dict, f)


def get_next_run_folder(runs_dir: Path, create: bool = True) -> Path:
    """Create a new run folder and return the path to it

    Args:
        runs_dir (Path): The directory of training runs
        create (bool): Whether to create the new run folder. Defaults to True

    Returns:
        Path: The path of the new run folder
    """
    # get a list of all folders in the root directory
    folders = [f for f in runs_dir.iterdir() if f.is_dir()]

    # filter the list to only include folders that match the run_XXX pattern
    run_folders = [f for f in folders if f.name.startswith("run_") and f.name[4:].isdigit()]

    # sort the list of run folders using natsort
    run_folders = natsort.natsorted(run_folders, key=lambda x: int(x.name[4:]))

    # if there are no run folders, create the first one
    if not run_folders:
        next_run_folder = runs_dir / "run_001"

    else:
        # get the last run folder and extract its number
        last_run_folder = run_folders[-1]
        last_run_number = int(last_run_folder.name[4:])

        # create the next run folder
        next_run_number = last_run_number + 1
        next_run_folder = runs_dir / f"run_{next_run_number:03d}"

    # create the next run folder if it doesn't exist
    if create:
        next_run_folder.mkdir(exist_ok=True)

    return next_run_folder
