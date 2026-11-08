import json
import logging
import sys
from pathlib import Path
from argparse import ArgumentParser
from clarity.utils.cli import (
    parse_int_or_none,
    parse_kwargs_into_dict,
    parse_casefold,
    parse_path,
    parse_truthy,
)
from clarity.utils.cli import add_standard_aliases
from clarity.utils.runtime import parse_weights
from clarity.utils.paths import resolve_cxr14_root, resolve_models_root
from clarity.training.config import TrainConfig
from clarity.data import splits as split_manifest
from clarity.training.setup import run_training
from clarity.training.sweep_spec import build_localization_sweep_spec  # noqa: F401  (test re-export)

"""Train a CLARiTy or ConvNeXtV2 model.

Usage::

    clarity-train [model_name] [model_size] [options]
"""


def _explicit_dests(parser, argv: list[str]) -> set[str]:
    """Destinations explicitly present on the command line (not defaults)."""
    dests = set()
    for token in argv:
        action = parser._option_string_actions.get(token.split("=", 1)[0])
        if action is not None:
            dests.add(action.dest)
    return dests


def _resolve_config_weights(config, *, dry_run: bool):
    """Resolve the configured init label, without loading it on ``--dry-run``."""
    value = config.model.init or config.model.weights or "imagenet"
    if dry_run:
        return (value, str(value))
    return parse_weights(value)


def _apply_train_config(args, parser, argv: list[str], *, dry_run: bool = False):
    """Fill CLI values from ``--config``/``--set`` without overriding flags."""
    if args.config is None and not args.config_overrides:
        return args
    if args.config is not None:
        config = TrainConfig.from_yaml(args.config)
    else:
        config = TrainConfig()
    if args.config_overrides:
        config = config.with_overrides(args.config_overrides)

    specified = _explicit_dests(parser, argv)
    values = {
        "model_name": config.model.name,
        "model_size": config.model.size,
        "size": config.data.image_size,
        "dataset": [config.data.dataset],
        "learning_rate": config.optim.learning_rate,
        "weight_decay": config.optim.weight_decay,
        "batch_size": config.optim.batch_size,
        "num_epochs": config.optim.num_epochs,
        "warmup_steps": config.optim.warmup_steps,
        "accumulation_factor": config.optim.accumulation_factor,
        "optimizer": config.optim.optimizer,
        "loss": [config.loss.name],
        "loss_kwargs": dict(config.loss.kwargs),
        "bbox_method": config.localization.bbox_method,
        "selection_strategy": config.localization.selection,
        "localization_optimize_separately": config.localization.optimize_metrics_separately,
        "localization_fit_workers": config.localization.fit_workers,
        "localization_attention_layer_sweep": (list(config.sweep.attention_layer_counts) or None),
        "localization_attention_primary": config.sweep.primary,
        "seed": config.seed,
        "run_folder": (Path(config.run_folder) if config.run_folder else None),
        "augment": (config.augmentation if config.augmentation is not None else False),
        "augmentation_kwargs": dict(config.augmentation_kwargs or {}),
    }
    # An explicit --weights/-wg must win without touching the configured
    # checkpoint: the 512 recipe points at a relative warm-start path that
    # only exists when that file was actually placed there.
    if "weights" not in specified:
        values["weights"] = _resolve_config_weights(config, dry_run=dry_run)
    if config.data.split:
        values["split"] = config.data.split
    if config.data.train_localization is not None:
        values["train_localization"] = config.data.train_localization
    if config.data.validate_localization is not None:
        values["validate_localization"] = config.data.validate_localization
    if config.data.border_padding is not None:
        values["border_padding"] = config.data.border_padding
    if config.model.kwargs:
        values["model_kwargs"] = dict(config.model.kwargs)
    if config.localization.method:
        values["localization_kwargs"] = {
            **dict(values.get("localization_kwargs", {})),
            "localization_method": config.localization.method,
        }
    if config.localization.attention_type:
        values["localization_kwargs"] = {
            **dict(values.get("localization_kwargs", {})),
            "inference_attention_type": config.localization.attention_type,
        }
    if config.data.images:
        values["images_path"] = [Path(config.data.images).expanduser()]
    if config.data.segmentations:
        values["segmentation_path"] = [Path(config.data.segmentations).expanduser()]
    if config.data.data_root:
        values["data_root"] = Path(config.data.data_root).expanduser()
    for dest, value in values.items():
        if dest not in specified:
            setattr(args, dest, value)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # set up argument parsing
    parser = ArgumentParser(description="Train a vision model on the given dataset/s.")
    parser.add_argument(
        "model_name", nargs="?", type=parse_casefold, default=None, help="Model to train"
    )
    parser.add_argument(
        "model_size", nargs="?", type=parse_casefold, default=None, help="Model size"
    )
    parser.add_argument(
        "-img", "--images_path", nargs="+", type=parse_path, default=None, help="Path/s to images"
    )
    parser.add_argument(
        "-seg",
        "--segmentation_path",
        nargs="+",
        type=parse_path,
        default=None,
        help="Path/s to segmentations",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        nargs="+",
        type=parse_casefold,
        default=["chestxray14"],
        help="Training dataset/s (default 'chestxray14')",
    )
    parser.add_argument("-s", "--size", type=int, default=224, help="Image size (default 224)")
    parser.add_argument(
        "-d",
        "--device",
        type=parse_casefold,
        default="cuda:0",
        help="The device used to train the model (default 'cuda:0')",
    )
    parser.add_argument(
        "-amp",
        "--use_amp",
        type=parse_truthy,
        default=False,
        help="Use bfloat16 autocast for model forward passes and loss computation (default False)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=parse_truthy,
        default=False,
        help="Whether to train in parallel on multiple GPUs (default False)",
    )
    parser.add_argument(
        "-c",
        "--jit_compile",
        type=parse_truthy,
        default=True,
        help="Whether to JIT-compile the model (default True)",
    )
    parser.add_argument(
        "-n",
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs to train for (default 100)",
    )
    parser.add_argument(
        "-esn",
        "--early_stop_num_epochs",
        type=parse_int_or_none,
        default=None,
        help="Number of epochs with no improvement resulting in an early stop (default inf)",
    )
    parser.add_argument(
        "-tn",
        "--terminate_epoch",
        type=parse_int_or_none,
        default=None,
        help="Exact epoch number to terminate at (default inf)",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size (default 64)")
    parser.add_argument(
        "-a",
        "--accumulation_factor",
        type=int,
        default=1,
        help="Accumulation factor. Defaults to 1 (update model weights every batch)",
    )
    parser.add_argument(
        "-wu", "--warmup_steps", type=int, default=5, help="Number of warmup steps (default 5)"
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        type=parse_casefold,
        default="adamw",
        help="The optimizer to use (default 'adamw')",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (Default 1e-3)",
    )
    parser.add_argument(
        "-dec", "--weight_decay", type=float, default=0.05, help="Weight decay (default 0.05)"
    )
    parser.add_argument(
        "-l",
        "--loss",
        nargs="+",
        type=parse_casefold,
        default=["wbce"],
        help="Loss function/s (default 'wbce')",
    )
    parser.add_argument(
        "-p1",
        "--split_pct1",
        type=float,
        default=0.8,
        help="The training split percent (default 0.8)",
    )
    parser.add_argument(
        "-p2",
        "--split_pct2",
        type=float,
        default=1.0,
        help="The training and validation split percent (default 1.0)",
    )
    parser.add_argument(
        "-spl",
        "--split",
        type=parse_casefold,
        default="random",
        help="Name of the evaluation split to use (ignores -p2, -p2; default 'random')",
    )
    parser.add_argument(
        "-wt",
        "--workers_train",
        type=int,
        default=16,
        help="Number of training dataloader workers (default 16)",
    )
    parser.add_argument(
        "-wv",
        "--workers_validate",
        type=int,
        default=16,
        help="Number of validation dataloader workers (default 16)",
    )
    parser.add_argument(
        "-wmp",
        "--mp_workers",
        type=int,
        default=16,
        help="Number of multiprocessing workers (default 16)",
    )
    parser.add_argument(
        "-mt",
        "--load_memory_train",
        type=parse_truthy,
        default=False,
        help="Whether to load all training images into memory",
    )
    parser.add_argument(
        "-mv",
        "--load_memory_validate",
        type=parse_truthy,
        default=False,
        help="Whether to load all validation images into memory",
    )
    parser.add_argument(
        "-wg",
        "--weights",
        type=parse_weights,
        default=("imagenet", "imagenet"),
        help="Initial weights label or weights Path to use for the model (default 'imagenet')",
    )
    parser.add_argument(
        "-mkw",
        "--model_kwargs",
        type=parse_kwargs_into_dict,
        default={},
        help="Model keyword arguments as 'key1=value1;key2=value2'. Values are parsed as Python literals (default {})",
    )
    parser.add_argument(
        "-lkw",
        "--loss_kwargs",
        type=parse_kwargs_into_dict,
        default={},
        help="Loss function keyword arguments as 'key1=value1;key2=value2'. Values are parsed as Python literals (default {})",
    )
    parser.add_argument(
        "-aug",
        "--augment",
        type=parse_truthy,
        default=False,
        help="Whether to augment the training images (default False)",
    )
    parser.add_argument(
        "-augkw",
        "--augmentation_kwargs",
        type=parse_kwargs_into_dict,
        default={},
        help="Augmentation keyword arguments as 'key1=value1;key2=value2' "
        "(default: the canonical augmentation settings)",
    )
    parser.add_argument(
        "-bg",
        "--pred_background",
        type=parse_truthy,
        default=False,
        help="Whether to make the model predict background (default False)",
    )
    parser.add_argument(
        "-bp",
        "--border_padding",
        type=int,
        default=0,
        help="Pad images and segmentation maps with a border of this size (default 0)",
    )
    parser.add_argument(
        "-tl",
        "--train_localization",
        type=parse_truthy,
        default=False,
        help="Train the model with segmentation maps (default False)",
    )
    parser.add_argument(
        "-seg_s",
        "--segm_size",
        type=parse_int_or_none,
        default=None,
        help="Resized segmentation map size (default None = the model's input image size)",
    )
    parser.add_argument(
        "-seg_dk",
        "--segm_dilate_kernel",
        type=int,
        default=0,
        help="Dilation of segmentation map kernel size (default 0)",
    )
    parser.add_argument(
        "-seg_di",
        "--segm_dilate_iterations",
        type=int,
        default=0,
        help="Dilation of segmentation map iterations (default 0)",
    )
    parser.add_argument(
        "-vl",
        "--validate_localization",
        type=parse_truthy,
        default=True,
        help="Whether to evaluate the model's localization performance during training (default True)",
    )
    parser.add_argument(
        "-lckw",
        "--localization_kwargs",
        type=parse_kwargs_into_dict,
        default={},
        help="Localization keyword arguments as 'key1=value1;key2=value2'. Values are parsed as Python literals (default {})",
    )
    parser.add_argument(
        "-bm",
        "--bbox_method",
        type=parse_casefold,
        default="onebox",
        choices=("singlecut", "onebox"),
        help="Bounding-box extraction method: 'onebox' (default; exact 0.05 "
        "grid + joint_argmax) or 'singlecut' (legacy). "
        "Default 'onebox'.",
    )
    parser.add_argument(
        "-sel",
        "--selection_strategy",
        type=parse_casefold,
        default=None,
        choices=("joint_argmax", "mean_argmax"),
        help="Calibration selection: 'joint_argmax' (corrected; default for "
        "onebox) or 'mean_argmax' (legacy; default for singlecut). "
        "Default None = method default.",
    )
    parser.add_argument(
        "-lsep",
        "--localization_optimize_separately",
        type=parse_truthy,
        default=False,
        help="Legacy: optimize IoBB and IoU independently (the historical "
        "two-call sequence). Only valid with -bm singlecut -sel "
        "mean_argmax; the default is one val5-IoU optimization "
        "reported for both families (default False).",
    )
    parser.add_argument(
        "-fw",
        "--localization_fit_workers",
        type=int,
        default=None,
        help="Fork-pool workers for the OneBox fit (default: min(-wmp, "
        "cpu_count)). Never changes the fit result.",
    )
    parser.add_argument(
        "-ptlp",
        "--precomputed_teacher_labels_path",
        nargs="+",
        type=parse_path,
        default=[None],
        help="Path for teacher labels to use for distillation (default None for training on grouth truth labels)",
    )
    parser.add_argument(
        "-i", "--info", type=str, default="", help="Information for logging (default '')"
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=None,
        help="Global RNG seed (default 0, the historic implicit value). Set "
        "before any model/dataset/dataloader construction so init "
        "weights, augmentation, and shuffling all inherit it.",
    )
    parser.add_argument(
        "-rf",
        "--run_folder",
        type=parse_path,
        default=None,
        help="Explicit run folder (default: auto-increment run_XXX in the model dir). "
        "Relative paths are resolved against the model directory.",
    )
    parser.add_argument(
        "-las",
        "--localization_attention_layer_sweep",
        nargs="+",
        type=int,
        default=None,
        metavar="COUNT",
        help="Enable the exact localization attention-layer-count sweep over these "
        "layer counts, e.g. '-las 1 2 3 4 5 6 7 8 9 10 11 12'. Requires "
        "localization validation and a CLARiTy model (default: no sweep)",
    )
    parser.add_argument(
        "-lasp",
        "--localization_attention_primary",
        type=int,
        default=None,
        metavar="COUNT",
        help="Primary attention-layer count of the sweep (default None; required with "
        "-las, and only allowed together with -las). Must be one of the -las "
        "counts; the recommended value is 5 for CLARiTy-S-16-224",
    )

    parser.add_argument(
        "--config",
        type=parse_path,
        default=None,
        help="YAML run configuration (CLI flags override it)",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="Override one resolved config field (repeatable)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved configuration and exit"
    )
    parser.add_argument(
        "--data-root",
        type=parse_path,
        default=None,
        help="Dataset root (overrides CLARITY_DATA_ROOT)",
    )
    parser.add_argument(
        "--models-root",
        type=parse_path,
        default=None,
        help="Run-folder root (default CLARITY_MODELS_ROOT or ~/models)",
    )
    parser.add_argument(
        "--splits-dir",
        type=parse_path,
        default=None,
        help="Directory with the four historical split lists",
    )
    parser.add_argument(
        "--verify-splits", action="store_true", help="Validate the supplied split manifest and exit"
    )
    add_standard_aliases(parser)

    # parse command line arguments
    args = parser.parse_args()
    args = _apply_train_config(args, parser, sys.argv[1:], dry_run=args.dry_run)
    if args.early_stop_num_epochs is None:
        args.early_stop_num_epochs = float("inf")
    if args.terminate_epoch is None:
        args.terminate_epoch = float("inf")
    if args.dry_run:
        print(json.dumps(vars(args), default=str, indent=2, sort_keys=True))
        raise SystemExit(0)

    split_manifest_data = (
        split_manifest.load_manifest(args.splits_dir) if args.splits_dir is not None else None
    )
    if args.verify_splits:
        if split_manifest_data is None:
            raise SystemExit("--verify-splits requires --splits-dir")
        split_manifest_data.validate_counts(split_manifest.PAPER_COUNTS)
        print(json.dumps(split_manifest_data.counts, indent=2))
        raise SystemExit(0)

    if split_manifest_data is not None and not (
        args.validate_localization or args.split.startswith("classify")
    ):
        print(
            "WARNING: --splits-dir is supplied but neither the classification "
            "split nor localization validation uses it (split="
            f"{args.split!r}, --validate-localization={args.validate_localization})",
            file=sys.stderr,
        )

    if args.model_name is None or args.model_size is None:
        parser.error("model_name and model_size are required (positionally or via --config)")
    cxr14_root = resolve_cxr14_root(args.data_root)
    models_root = (
        Path(args.models_root).expanduser()
        if args.models_root is not None
        else resolve_models_root()
    )
    if args.images_path is None:
        args.images_path = [cxr14_root]
    if args.segmentation_path is None:
        args.segmentation_path = [cxr14_root]
    if args.precomputed_teacher_labels_path is None:
        args.precomputed_teacher_labels_path = [None]

    run_training(
        args,
        cxr14_root=cxr14_root,
        models_root=models_root,
        split_manifest_data=split_manifest_data,
    )


if __name__ == "__main__":
    main()
