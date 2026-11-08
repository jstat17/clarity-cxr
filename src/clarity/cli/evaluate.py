import json
import logging
import sys
from pathlib import Path
from argparse import ArgumentParser
from clarity.utils.cli import add_standard_aliases
from clarity.utils.runtime import parse_weights
from clarity.utils.cli import (
    parse_casefold,
    parse_int_or_none,
    parse_kwargs_into_dict,
    parse_path,
    parse_truthy,
)
from clarity.utils.paths import resolve_cxr14_root, resolve_rsna_root, resolve_vindr_root
from clarity.evaluation.arms import ARM_A1, ARM_A2, ARM_IDS, ARM_NAMES, arm_label
from clarity.localization.artifacts import load_frozen_calibration
from clarity.training.config import EvalConfig
from clarity.evaluation.setup import run_evaluation_setup
from clarity.data import splits as split_manifest

"""Evaluate a CLARiTy model (classification or localization).

Usage::

    clarity-evaluate [model_name] [model_size] --split classify|localize [options]
"""
#: ``--arm`` text (short id or paper name, casefolded) -> arm id
_ARM_BY_TEXT = {
    **{arm.casefold(): arm for arm in ARM_NAMES},
    **{name.casefold(): arm for name, arm in ARM_IDS.items()},
}
LOCALIZATION_TEST_SCORE_THRESHOLDS = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
]


def _explicit_dests(parser, argv: list[str]) -> set[str]:
    dests = set()
    for token in argv:
        action = parser._option_string_actions.get(token.split("=", 1)[0])
        if action is not None:
            dests.add(action.dest)
    return dests


def _apply_eval_config(args, parser, argv: list[str], *, dry_run: bool = False):
    """Fill evaluation CLI values from ``--config``/``--set``."""
    if args.config is None and not args.config_overrides:
        return args
    config = EvalConfig.from_yaml(args.config) if args.config is not None else EvalConfig()
    if args.config_overrides:
        config = config.with_overrides(args.config_overrides)
    specified = _explicit_dests(parser, argv)
    values = {
        "split": config.split,
        "model_name": config.model.name,
        "model_size": config.model.size,
        "num_classes": config.model.num_classes,
        "size": config.data.image_size,
        "dataset": [config.data.dataset],
        "bbox_method": config.localization.bbox_method,
        "selection_strategy": config.localization.selection,
        "localization_optimize_separately": config.localization.optimize_metrics_separately,
        "localization_fit_workers": config.localization.fit_workers,
    }
    if config.model.kwargs:
        values["model_kwargs"] = dict(config.model.kwargs)
    # An explicit --weights/-wg wins without resolving a configured init or
    # checkpoint path (which may deliberately be a placeholder).
    if "weights" not in specified:
        init = config.model.init or config.model.weights
        if init:
            values["weights"] = (init, str(init)) if dry_run else parse_weights(init)
    if config.localization.method:
        values["localization_kwargs"] = {"localization_method": config.localization.method}
    if config.localization.attention_type:
        values["localization_kwargs"] = {
            **values.get("localization_kwargs", {}),
            "inference_attention_type": config.localization.attention_type,
        }
    if config.data.images:
        values["images_path"] = [Path(config.data.images).expanduser()]
    if config.data.segmentations:
        values["segmentation_path"] = [Path(config.data.segmentations).expanduser()]
    if config.data.data_root:
        values["data_root"] = Path(config.data.data_root).expanduser()
    if config.models_root:
        values["models_root"] = Path(config.models_root).expanduser()
    for dest, value in values.items():
        if dest not in specified:
            setattr(args, dest, value)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # set up argument parsing
    parser = ArgumentParser(description="Evaluate a vision model on the given dataset/s.")
    parser.add_argument(
        "model_name", nargs="?", type=parse_casefold, default=None, help="Model to evaluate"
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
        help="Evaluation dataset/s (default 'chestxray14')",
    )
    parser.add_argument(
        "-cls",
        "--num_classes",
        type=parse_int_or_none,
        default=None,
        help="Number of classes, if not specified defaults to the number of "
        "dataset labels. The transfer arms A1/A2 evaluate the 14-class NIH "
        "checkpoint, so they pass 14 (default None)",
    )
    parser.add_argument("-s", "--size", type=int, default=224, help="Image size (default 224)")
    parser.add_argument(
        "-d",
        "--device",
        type=parse_casefold,
        default="cuda:0",
        help="The device used to evaluate the model (default 'cuda:0')",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=parse_truthy,
        default=True,
        help="Whether to evaluate in parallel on multiple GPUs (default True)",
    )
    parser.add_argument(
        "-c",
        "--jit_compile",
        type=parse_truthy,
        default=True,
        help="Whether to JIT-compile the model (default True)",
    )
    parser.add_argument(
        "-amp",
        "--use_amp",
        type=parse_truthy,
        default=False,
        help="Use bfloat16 autocast for model forward passes and loss computation (default False)",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size (default 64)")
    parser.add_argument(
        "-l", "--loss", type=parse_casefold, default="wbce", help="Loss function (default 'wbce')"
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
        default="classify",
        choices=("classify", "localize"),
        help="Evaluation split: 'classify' (default) or 'localize'",
    )
    parser.add_argument(
        "-we",
        "--workers_evaluate",
        type=int,
        default=8,
        help="Number of validation dataloader workers (default 8)",
    )
    parser.add_argument(
        "-wmp",
        "--mp_workers",
        type=int,
        default=16,
        help="Number of multiprocessing workers (default 16)",
    )
    parser.add_argument(
        "-wg",
        "--weights",
        type=parse_weights,
        default=("imagenet", "imagenet"),
        help="Weights label or weights Path to use for the model (default 'imagenet')",
    )
    parser.add_argument(
        "-mkw",
        "--model_kwargs",
        type=parse_kwargs_into_dict,
        default={},
        help="Model keyword arguments as 'key1=value1;key2=value2'. Values are parsed as Python literals (default {})",
    )
    parser.add_argument(
        "-alc",
        "--attention_layer_count",
        type=int,
        default=None,
        help="Explicit localization attention-layer count override (CLARiTy model_attention localization). Takes precedence over the run's best-selection.json manifest and over a legacy n_attn_layers_avg model/localization kwarg; for sweep runs the documented in-sweep values are 2/4/6/8/10/12, but any count up to model depth is accepted (default: use the manifest-selected count when evaluating a sweep run's model_best.pth, otherwise the model/default n_attn_layers_avg)",
    )
    parser.add_argument(
        "-lkw",
        "--loss_kwargs",
        type=parse_kwargs_into_dict,
        default={},
        help="Loss function keyword arguments as 'key1=value1;key2=value2'. Values are parsed as Python literals (default {})",
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
        "-ioutm",
        "--iou_thr_merge",
        type=float,
        default=1.0,
        help="IoU threshold between pairs of predicted bounding boxes to cause a merge (default 1.0)",
    )
    parser.add_argument(
        "-proxt",
        "--proximity_thr",
        type=float,
        default=0.0,
        help="Euclidean distance threshold between pairs of predicted bounding boxes to cause a merge (in pixels, default 0)",
    )
    parser.add_argument(
        "-occ",
        "--only_correctly_classified",
        type=parse_truthy,
        default=False,
        help="Whether to use only correctly-classified images for localization metrics (default False; localization uses all images)",
    )
    parser.add_argument(
        "-sf",
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to the end of the evaluation log file (default '')",
    )

    parser.add_argument(
        "--config", type=parse_path, default=None, help="YAML evaluation configuration"
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
    parser.add_argument("--models-root", type=parse_path, default=None, help="Run-folder root")
    parser.add_argument(
        "--splits-dir",
        type=parse_path,
        default=None,
        help="Directory with the four historical split lists",
    )
    parser.add_argument(
        "--verify-splits", action="store_true", help="Validate the supplied split manifest and exit"
    )
    parser.add_argument(
        "--rsna-splits-dir",
        type=parse_path,
        default=None,
        help="Directory with the three RSNA split lists (rsna_{train,val,test}_list.txt, "
        "one image filename per line); required for -ds rsna",
    )
    parser.add_argument(
        "--vindr-splits-dir",
        type=parse_path,
        default=None,
        help="Directory with the three VinDr-CXR split lists (vindr_{train,val,test}_list.txt, "
        "one image filename per line); required for -ds vindr",
    )
    parser.add_argument(
        "--vindr-fuse-outputs",
        action="store_true",
        help="VinDr only: fuse the model's Mass/Nodule outputs into one Nodule/Mass "
        "finding and select the target label space before scoring. Required for "
        "arms A1 (NIH-calibrated) and A2 (Target-calibrated), which evaluate the "
        "14-class NIH checkpoint behind a logit_mask; never implicit",
    )
    parser.add_argument(
        "--frozen-onebox-calibration",
        type=parse_path,
        default=None,
        help="JSON file with a schema-1 'localization_method' block (the shipped "
        "calibration/a1_onebox_nih_*.json). The validation pass is then scored with "
        "the ported parameters instead of being refitted: arm A1, NIH-calibrated "
        "(strict zero-shot). Omit it to refit on the target validation split: "
        "arm A2, Target-calibrated",
    )
    parser.add_argument(
        "--arm",
        type=parse_casefold,
        default=None,
        choices=sorted(_ARM_BY_TEXT),
        help="Optional guard naming the transfer arm: 'a1' / 'nih-calibrated' "
        "requires --frozen-onebox-calibration, 'a2' / 'target-calibrated' forbids it",
    )
    add_standard_aliases(parser)

    # parse command line arguments
    args = parser.parse_args()
    args = _apply_eval_config(args, parser, sys.argv[1:], dry_run=args.dry_run)
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
    if args.model_name is None or args.model_size is None:
        parser.error("model_name and model_size are required (positionally or via --config)")

    if args.frozen_onebox_calibration is not None and args.split != "localize":
        parser.error("--frozen-onebox-calibration requires '--split localize'")
    if args.arm is not None:
        arm = _ARM_BY_TEXT[args.arm]
        if arm == ARM_A1 and args.frozen_onebox_calibration is None:
            parser.error(
                f"arm {arm_label(ARM_A1)} needs --frozen-onebox-calibration "
                "(see calibration/ in the repository)"
            )
        if arm == ARM_A2 and args.frozen_onebox_calibration is not None:
            parser.error(
                f"arm {arm_label(ARM_A2)} refits on the target validation split; "
                "drop --frozen-onebox-calibration"
            )

    # resolve only the roots the selected dataset needs
    dataset = args.dataset[0].casefold()
    cxr14_root = resolve_cxr14_root(args.data_root) if dataset == "chestxray14" else None
    rsna_root = resolve_rsna_root(args.data_root) if dataset == "rsna" else None
    vindr_root = resolve_vindr_root(args.data_root) if dataset == "vindr" else None
    try:
        frozen_calibration = load_frozen_calibration(args.frozen_onebox_calibration)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if dataset == "chestxray14":
        if args.images_path is None:
            args.images_path = [cxr14_root]
        if args.segmentation_path is None:
            args.segmentation_path = [cxr14_root]
    else:
        # RSNA/VinDr images are converted PNGs and the anatomical masks are
        # generated; there is no default location for either
        if args.images_path is None:
            parser.error(f"-ds {dataset} needs -img/--images_path (the converted PNG directory)")
        if args.segmentation_path is None:
            if args.split == "localize":
                parser.error(
                    f"-ds {dataset} --split localize needs -seg/--segmentation_path "
                    "(the anatomical masks from clarity-prepare)"
                )
            args.segmentation_path = [None]

    run_evaluation_setup(
        args,
        cxr14_root=cxr14_root,
        split_manifest_data=split_manifest_data,
        rsna_root=rsna_root,
        rsna_splits_dir=args.rsna_splits_dir,
        vindr_root=vindr_root,
        vindr_splits_dir=args.vindr_splits_dir,
        frozen_calibration=frozen_calibration,
    )


if __name__ == "__main__":
    main()
