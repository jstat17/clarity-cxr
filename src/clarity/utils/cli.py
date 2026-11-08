"""Argument parsers and value coercions shared by the CLIs."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any
from collections.abc import Iterable


def parse_kwargs_into_dict(s: str) -> dict[str, Any]:
    """Parse a string of key1=value1;key2=value2 list of pairs into a dictionary.

    Args:
        s (str): Key word arguments encoded as a string, format: "key1=value1;..."

    Raises:
        argparse.ArgumentTypeError: Must be in the correct format
        argparse.ArgumentTypeError: Must be a python literal

    Returns:
        dict[str, Any]: Key word arguments
    """
    if not s.strip():
        return {}

    kwargs = {}
    for part in s.split(";"):
        key, _, value_str = part.partition("=")

        if not key or not value_str:
            raise argparse.ArgumentTypeError(f"Invalid format: '{part}'. Use 'key=value'.")

        try:
            value = ast.literal_eval(value_str.strip())

        except (ValueError, SyntaxError) as e:
            raise argparse.ArgumentTypeError(f"Invalid value '{value_str}' for key '{key}': {e}")

        kwargs[key.strip()] = value

    return kwargs


def parse_int_or_none(s: str) -> int | None:
    """Parse a string into an int or None.

    Args:
        s (str): An integer, or 'None'/'none' for no value

    Raises:
        argparse.ArgumentTypeError: Must be an integer or 'None'

    Returns:
        int | None: Parsed integer, or None
    """
    if s.casefold() == "none":
        return None

    try:
        return int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{s}'. Must be an integer or 'None'"
        ) from e


def parse_truthy(s: str) -> bool:
    """Parse a string into a bool.

    Args:
        s (str): 'True'/'Yes' or 'False'/'No', any case

    Raises:
        argparse.ArgumentTypeError: Must be 'True'/'Yes' or 'False'/'No'

    Returns:
        bool: Argument truthy value
    """
    lowered = s.strip().casefold()
    if lowered in ("true", "t", "yes", "y", "1", "on"):
        return True
    if lowered in ("false", "f", "no", "n", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid value '{s}'. Must be one of True/False/Yes/No/1/0")


def add_standard_aliases(parser: argparse.ArgumentParser) -> None:
    """Add hidden kebab-case and ``--no-*`` aliases to an existing parser.

    The historical CLI uses underscores and single-dash short options.
    The public flags are the kebab-case long forms; the old spellings stay as
    hidden aliases so older invocations keep working for one release.
    """
    for action in list(parser._actions):
        if not action.option_strings or action.dest in (argparse.SUPPRESS, "help"):
            continue
        for option in list(action.option_strings):
            if not option.startswith("--") or "_" not in option:
                continue
            alias = "--" + option[2:].replace("_", "-")
            if alias in parser._option_string_actions:
                continue
            kwargs = dict(
                dest=action.dest,
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
                type=action.type,
                nargs=action.nargs,
                choices=action.choices,
            )
            parser.add_argument(alias, **kwargs)
        if action.type is parse_truthy:
            for option in action.option_strings:
                if not option.startswith("--"):
                    continue
                stem = option[2:].replace("_", "-")
                no_flag = f"--no-{stem}"
                if no_flag in parser._option_string_actions:
                    continue
                parser.add_argument(
                    no_flag,
                    dest=action.dest,
                    action="store_false",
                    default=argparse.SUPPRESS,
                    help=argparse.SUPPRESS,
                )


def parse_casefold(s: str | Iterable[str]) -> str | list[str]:
    """Parse a string or iterable of strings into casefolded string/s

    Args:
        s (str | Iterable[str]): String/s

    Returns:
        str | list[str]: Casefolded string/s
    """
    if isinstance(s, str):
        s = s.casefold()

    elif isinstance(s, Iterable):
        s = [sub.casefold() for sub in s]
        if len(s) == 1:
            s = s[0]

    return s


def parse_path(s: str | Iterable[str]) -> Path | list[Path]:
    """Parse a string or iterable of strings into Path object/s

    Args:
        s (str | Iterable[str]): Path-like string/s

    Returns:
        Path: Expanded Path object/s
    """
    if isinstance(s, str):
        pth = Path(s).expanduser()

    elif isinstance(s, Iterable):
        pth = [Path(sub).expanduser() for sub in s]
        if len(pth) == 1:
            pth = pth[0]

    return pth
