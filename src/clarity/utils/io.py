"""Run-folder naming, serialization and command-line recording helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np


def create_heatmaps_folder_name(
    checkpoint_name: str, dataset: str, heatmap_type: str, **kwargs: dict[str, Any]
) -> str:
    """Create a folder name in the form:
    checkpoint_name-dataset-heatmap_type;kwarg1=val1-kwarg2=val2...

    Args:
        checkpoint_name (str): Name of the trained model checkpoint
        dataset (str): Name of the dataset
        heatmap_type (str): Heatmap type
        kwargs (dist[str, Any]): Key words of heatmap options

    Returns:
        str: Folder name combining all arguments
    """
    folder_name = "-".join([checkpoint_name, dataset, heatmap_type])

    if kwargs is not None:
        folder_kwargs = ["=".join([key, str(value)]) for key, value in kwargs.items()]
        folder_kwargs = "-".join(folder_kwargs)

        folder_name += ";" + folder_kwargs

    return folder_name


def get_terminal_command() -> str:
    """Get the command-line invocation of the current script.

    Note: This method uses /proc/self/cmdline, which is only available on Linux.
    The output reflects the arguments after shell parsing; quotes and other formatting
    are not preserved.

    Returns:
        str: Terminal command after shell parsing
    """
    # read the command-line arguments from /proc/self/cmdline
    with open("/proc/self/cmdline", "rb") as f:
        content = f.read()

    # split the content by the null byte which separates arguments
    args = content.split(b"\0")

    # remove the empty final element if present
    if args and args[-1] == b"":
        args = args[:-1]

    # decode each argument from bytes to str
    decoded_args = [arg.decode("utf-8", errors="replace") for arg in args]

    # join with a space to form a single string
    command = " ".join(decoded_args)

    return command


def serialize_data(data: Any) -> Any:
    """
    Recursively converts NumPy arrays and scalar types in a nested dictionary or array to
    JSON-serializable formats. Handles NumPy types in both values and dictionary keys.

    Args:
        data (Any): A NumPy array, scalar, or dictionary that may contain nested dictionaries,
        arrays, or NumPy-typed keys.

    Returns:
        Any: A JSON-serializable structure with NumPy arrays converted to lists and NumPy keys
        converted to strings.

    Raises:
        TypeError: If an unsupported type is encountered that cannot be serialized to JSON.
    """
    # handle numpy arrays
    if isinstance(data, np.ndarray):
        return data.tolist()

    # handle numpy scalar types (e.g., np.float64, np.int64)
    elif isinstance(data, (np.floating, np.integer)):
        return float(data) if isinstance(data, np.floating) else int(data)

    # handle dictionaries with potential numpy-typed keys
    elif isinstance(data, dict):
        return {
            (
                str(serialize_data(key))
                if isinstance(key, (np.ndarray, np.floating, np.integer))
                else key
            ): serialize_data(value)
            for key, value in data.items()
        }

    # handle lists or tuples
    elif isinstance(data, (list, tuple)):
        return [serialize_data(item) for item in data]

    # handle basic JSON-serializable types
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data

    else:
        raise TypeError(f"Object of type {type(data)} is not JSON serializable")


def convert_defaultdict_to_dict(d: dict | defaultdict):
    if isinstance(d, (defaultdict, dict)):
        return {key: convert_defaultdict_to_dict(value) for key, value in d.items()}

    elif isinstance(d, list):
        return [convert_defaultdict_to_dict(v) for v in d]

    return d


def clean_command(cmd: str) -> str:
    # strip trailing/leading whitespace
    cmd = cmd.strip()
    # replace multiple whitespace with a single space
    cmd = re.sub(r"\s+", " ", cmd)

    return cmd
