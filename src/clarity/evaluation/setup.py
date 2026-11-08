"""Evaluation setup: data, model, criterion and loader assembly.

The CLI module :mod:`clarity.cli.evaluate` only parses arguments and
delegates here.  Importable without side effects so tests can patch source
modules (loader, model factory, evaluators).
"""

from __future__ import annotations

import logging
import sys

from pathlib import Path

import torch as th

from clarity.data import chestxray14, rsna, vindr
from clarity.data import splits as split_manifest
from clarity.evaluation.arms import ARM_A1, ARM_A2, arm_label
from clarity.evaluation.loop import run_evaluation
from clarity.losses.registry import get_loss
from clarity.data import loader as loader_utils
from clarity.localization.sweep import (
    has_sweep_capability,
    localization_selection_metadata,
    unwrap_model,
)
from clarity.models import factory as vision_get
from clarity.models.fuse_findings import FindingOutputAdapter


logger = logging.getLogger(__name__)

#: datasets of the transfer study (the target of arms A1 and A2)
TRANSFER_DATASETS = ("rsna", "vindr")


def _report_vindr_split_coverage(
    splits_dir: Path, validate_filenames: list[str], test_filenames: list[str]
) -> None:
    """Say how much of each split list is actually on disk.

    Arm A1 (NIH-calibrated) can score a partial validation split; arm A2
    (Target-calibrated) refits on it and needs it complete, so the coverage has
    to be visible before the run rather than inferred from the numbers.
    """
    listed_val = vindr.count_split_filenames(splits_dir, "val")
    listed_test = vindr.count_split_filenames(splits_dir, "test")
    print(
        f"VinDr split coverage: val {len(validate_filenames)}/{listed_val}, "
        f"test {len(test_filenames)}/{listed_test}"
    )
    if len(validate_filenames) < listed_val:
        print(
            f"  note: {arm_label(ARM_A1)} scores its frozen parameters on the "
            f"available val images; {arm_label(ARM_A2)} needs the full val split "
            "before refitting"
        )


def _require_split_dir(dataset: str, splits_dir: Path | None) -> Path:
    if splits_dir is None:
        raise SystemExit(
            f"{dataset} needs --{dataset}-splits-dir: a directory with the three "
            "split lists (one image filename per line); see docs/transfer.md"
        )
    return splits_dir


def _vindr_split_filenames(splits_dir: Path | None, images_path: Path):
    splits_dir = _require_split_dir("vindr", splits_dir)
    train, validate, test = vindr.get_split_filenames(
        splits_dir=splits_dir, available_images_path=images_path
    )
    _report_vindr_split_coverage(splits_dir, validate, test)
    if not validate:
        raise SystemExit(
            "VinDr validation split is empty after filtering against "
            f"{images_path}; convert the validation images first. The evaluator "
            "always runs a validation pass, so it needs those images, not only the "
            "test set"
        )
    return train, validate, test


def run_evaluation_setup(
    args,
    *,
    cxr14_root: Path | None,
    split_manifest_data,
    rsna_root: Path | None = None,
    rsna_splits_dir: Path | None = None,
    vindr_root: Path | None = None,
    vindr_splits_dir: Path | None = None,
    frozen_calibration: dict | None = None,
) -> None:
    """Build the evaluation run from parsed args and execute it."""
    model_name = args.model_name
    model_size = args.model_size

    images_path = args.images_path[0]
    segmentation_path = args.segmentation_path[0]
    dataset = args.dataset[0]
    image_size = args.size

    device = args.device
    parallel = args.parallel
    jit_compile = args.jit_compile
    use_amp = args.use_amp

    batch_size = args.batch_size
    loss = args.loss

    split = args.split

    workers_evaluate = args.workers_evaluate
    mp_workers = args.mp_workers

    weights, weights_info = args.weights
    model_kwargs = args.model_kwargs
    loss_kwargs = args.loss_kwargs
    localization_kwargs = args.localization_kwargs

    explicit_attention_layer_count = args.attention_layer_count

    pred_background = args.pred_background
    border_padding = args.border_padding
    iou_thr_merge = args.iou_thr_merge
    proximity_thr = args.proximity_thr or None  # 0.0 means "no merging"
    only_correctly_classified = args.only_correctly_classified

    suffix = args.suffix

    # Explicit CLI controls take precedence over free-form localization kwargs.
    # Remove duplicate reserved keys before the final ``main`` call. A legacy
    # ``n_attn_layers_avg`` localization kwarg conflicts with the explicit
    # ``--attention_layer_count`` argument: the new argument wins and a clear
    # warning is printed; the model kwarg is never silently used.
    localization_kwargs = dict(localization_kwargs)
    legacy_n_attn_layers_avg = localization_kwargs.pop("n_attn_layers_avg", None)
    # A legacy ``n_attn_layers_avg`` model kwarg (-mkw) is still accepted for
    # building the model; if it conflicts with --attention_layer_count, the
    # explicit argument wins with a clear warning and the model kwarg never
    # selects the localization layer count.
    if legacy_n_attn_layers_avg is None:
        legacy_n_attn_layers_avg = model_kwargs.get("n_attn_layers_avg")
    if (
        explicit_attention_layer_count is not None
        and legacy_n_attn_layers_avg is not None
        and explicit_attention_layer_count != legacy_n_attn_layers_avg
    ):
        print(
            f"WARNING: conflicting localization attention-layer counts: "
            f"--attention_layer_count {explicit_attention_layer_count} wins "
            f"over the legacy model/localization kwarg n_attn_layers_avg="
            f"{legacy_n_attn_layers_avg}; the model kwarg is NOT used for "
            f"localization.",
            file=sys.stderr,
        )
    localization_kwargs.pop("only_correctly_classified", None)
    localization_kwargs.pop("use_amp", None)
    # Explicit bbox-method flags win over any -lckw duplicates so the method
    # is visible in the command, not inherited silently.
    localization_kwargs.pop("bbox_method", None)
    localization_kwargs.pop("selection", None)
    localization_kwargs.pop("optimize_metrics_separately", None)
    localization_kwargs.pop("fit_workers", None)
    bbox_method = args.bbox_method
    selection = args.selection_strategy
    optimize_metrics_separately = bool(args.localization_optimize_separately)
    fit_workers = args.localization_fit_workers

    # get dataset info
    match dataset:
        case "chestxray14":
            df = chestxray14.get_chestxray14_dataframe(cxr14_root)
            possible_labels = chestxray14.CXR_14_ABNORMALITIES
            label_anatomical_regions = chestxray14.CXR_14_ABNORMALITY_REGIONS
        case "rsna":
            df = rsna.get_rsna_dataframe(rsna_root)
            possible_labels = rsna.RSNA_DISEASES
            label_anatomical_regions = rsna.RSNA_DISEASE_REGIONS

        case "vindr":
            # the label space follows the split: 11 mapped findings for the
            # classification metrics, the 6 box findings for localization (the
            # OneBox outputs after the Nodule/Mass fusion)
            df = vindr.get_vindr_dataframe(vindr_root, "all")
            if split == "localize":
                possible_labels = vindr.VINDR_BOX_FINDINGS
                label_anatomical_regions = {
                    name: vindr.VINDR_DISEASE_REGIONS[name] for name in vindr.VINDR_BOX_FINDINGS
                }
            else:
                possible_labels = vindr.VINDR_DISEASES
                label_anatomical_regions = vindr.VINDR_DISEASE_REGIONS

        case _:
            raise SystemExit(
                f"unsupported dataset {dataset!r}; expected 'chestxray14', 'rsna' or 'vindr'"
            )

    requested_num_classes = getattr(args, "num_classes", None)
    num_classes = len(possible_labels) if requested_num_classes is None else requested_num_classes

    # A classification-only evaluation never needs the localization heads; set
    # ``only_classify`` here so a plain ``--split classify`` works. An explicit
    # model kwarg still wins.
    if split == "classify" and "only_classify" not in model_kwargs:
        model_kwargs = {**model_kwargs, "only_classify": True}

    # load model based on arguments
    model = vision_get.get_model(
        model_name=model_name,
        model_size=model_size,
        num_classes=num_classes,
        weights=weights,
        pred_background=pred_background,
        **model_kwargs,
    )

    # An explicit attention-layer count is only meaningful for CLARiTy
    # model_attention localization; reject it clearly for any other model.
    if explicit_attention_layer_count is not None:
        unwrapped_model = unwrap_model(model)
        if not has_sweep_capability(model):
            raise SystemExit(
                "--attention_layer_count requires a CLARiTy model with the "
                "localization attention-layer sweep capability; got model with "
                f"fullname {getattr(unwrapped_model, 'fullname', None)!r}"
            )
        if split == "localize":
            localization_method = str(localization_kwargs.get("localization_method", "cam"))
            if localization_method.casefold() != "model_attention":
                raise SystemExit(
                    "--attention_layer_count requires localization_method "
                    f"'model_attention' for localization evaluation; got "
                    f"{localization_method!r}"
                )

    # The evaluator's historical default is grad-CAM maps; make the resolved
    # method explicit so it cannot be dropped on the way to the evaluator.
    localization_kwargs.setdefault("localization_method", "model_attention")

    # VinDr's Nodule/Mass is one target finding but two NIH classes (arms A1 and
    # A2: the 14-class NIH checkpoint behind a ``logit_mask``). Derive the adapter
    # plan from the frozen mapping and refuse to run when the model's class axis
    # does not match the evaluation label space: silently indexing Mass as the
    # fused finding would report a plausible but wrong localization number.
    vindr_output_plan = None
    if dataset == "vindr":
        label_space = "localization" if split == "localize" else "classification"
        logit_mask = model_kwargs.get("logit_mask")
        if logit_mask is not None and not isinstance(logit_mask, (list, tuple)):
            raise SystemExit(
                "VinDr arms A1/A2 need an explicit list logit_mask "
                f"(got {type(logit_mask).__name__}); see docs/transfer.md"
            )
        vindr_output_plan = vindr.finding_output_plan(label_space, logit_mask=logit_mask)
        if vindr_output_plan.is_identity:
            if getattr(args, "vindr_fuse_outputs", False):
                raise SystemExit(
                    "--vindr-fuse-outputs was requested but this model already emits "
                    f"the {label_space} label space unchanged (num_classes={num_classes})"
                )
            vindr_output_plan = None
        else:
            if not getattr(args, "vindr_fuse_outputs", False):
                raise SystemExit(
                    f"VinDr {label_space} with num_classes={num_classes} emits "
                    f"{vindr_output_plan.emitted_width} classes but the label space has "
                    f"{len(vindr_output_plan.targets)}; pass --vindr-fuse-outputs to fuse "
                    "Nodule/Mass and select the target findings"
                )
            if split == "localize":
                method = str(localization_kwargs.get("localization_method", "cam")).casefold()
                if method not in ("model_attention", "cam"):
                    raise SystemExit(
                        "--vindr-fuse-outputs supports 'model_attention' and 'cam' "
                        f"localization on VinDr; got {method!r}"
                    )
            print(
                f"VinDr {label_space}: adapting {vindr_output_plan.emitted_width} model "
                f"outputs to {len(vindr_output_plan.targets)} findings "
                f"(fuse_pairs={vindr_output_plan.fuse_pairs}, keep={vindr_output_plan.keep})"
            )

    # Name the transfer arm in the log: the frozen NIH-fitted calibration is
    # arm A1, a calibration refitted on the target validation split is arm A2.
    if dataset in TRANSFER_DATASETS and split == "localize":
        arm = ARM_A1 if frozen_calibration is not None else ARM_A2
        print(f"Transfer arm: {arm_label(arm)} on {dataset}")

    # select dataset split
    match split:
        case "classify":
            return_filename = False
            df_bbox = None
            segmentation_path = None
            match dataset:
                case "chestxray14":
                    if split_manifest_data is not None:
                        validate_filenames = split_manifest.filter_filenames(
                            split_manifest_data.classification_val,
                            images_path,
                            chestxray14.IGNORED_IMAGES,
                        )
                    else:
                        _filenames, validate_filenames = chestxray14.get_classification_split_filenames(
                            chestxray14_path=cxr14_root,
                            available_images_path=images_path,
                            train_size=70,
                        )
                    _filenames, test_filenames = chestxray14.get_official_split_filenames(
                        chestxray14_path=cxr14_root, available_images_path=images_path
                    )

                case "rsna":
                    _train_filenames, validate_filenames, test_filenames = (
                        rsna.get_split_filenames(
                            splits_dir=_require_split_dir("rsna", rsna_splits_dir),
                            available_images_path=images_path,
                        )
                    )

                case "vindr":
                    _train_filenames, validate_filenames, test_filenames = (
                        _vindr_split_filenames(vindr_splits_dir, images_path)
                    )

                case _:
                    raise SystemExit(
                        f"classification evaluation is not implemented for {dataset!r}"
                    )

            # create validate dataset
            validate_dataset = loader_utils.MultilabelDataset(
                df=df,
                images_path=images_path,
                img_shape=image_size,
                split="validate",
                split_pct1=0.7,
                split_pct2=0.8,
                possible_labels=possible_labels,
                filenames=validate_filenames,
                use_background_label=pred_background,
                df_bbox=df_bbox,
                return_filename=return_filename,
                segmentation_path=segmentation_path,
                label_anatomical_regions=label_anatomical_regions,
                border_padding=border_padding,
            )
            validate_loader = th.utils.data.DataLoader(
                dataset=validate_dataset,
                batch_size=batch_size,
                num_workers=workers_evaluate,
                shuffle=False,
                collate_fn=loader_utils.custom_collate,
                worker_init_fn=loader_utils.dataloader_worker_init_fn,
            )

            # create test dataset
            test_dataset = loader_utils.MultilabelDataset(
                df=df,
                images_path=images_path,
                img_shape=image_size,
                split="test",
                split_pct1=0.8,
                split_pct2=1.0,
                possible_labels=possible_labels,
                filenames=test_filenames,
                use_background_label=pred_background,
                df_bbox=df_bbox,
                multi_box_gt=dataset in TRANSFER_DATASETS,
                return_filename=return_filename,
                segmentation_path=segmentation_path,
                label_anatomical_regions=label_anatomical_regions,
                border_padding=border_padding,
            )
            test_loader = th.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                num_workers=workers_evaluate,
                shuffle=False,
                collate_fn=loader_utils.custom_collate,
                worker_init_fn=loader_utils.dataloader_worker_init_fn,
            )

            evaluate_loader = [validate_loader, test_loader]

            # set loss function
            criterion = get_loss(
                loss_name=loss,
                pred_background=pred_background,
                labels=test_dataset.labels,
                device=device,
                **loss_kwargs,
            )

        case "localize":
            return_filename = True
            match dataset:
                case "chestxray14":
                    if split_manifest_data is not None:
                        validate_filenames = split_manifest.filter_filenames(
                            split_manifest_data.localization_val,
                            images_path,
                            chestxray14.IGNORED_IMAGES,
                        )
                        test_filenames = split_manifest.filter_filenames(
                            split_manifest_data.localization_test,
                            images_path,
                            chestxray14.IGNORED_IMAGES,
                        )
                    else:
                        validate_filenames, test_filenames = (
                            chestxray14.get_localization_split_filenames(
                                chestxray14_path=cxr14_root
                            )
                        )
                    df_bbox = chestxray14.get_chestxray14_bbox_dataframe(
                        chestxray14_path=cxr14_root, image_size=image_size
                    )

                case "rsna":
                    _train_filenames, validate_filenames, test_filenames = (
                        rsna.get_split_filenames(
                            splits_dir=_require_split_dir("rsna", rsna_splits_dir),
                            available_images_path=images_path,
                        )
                    )
                    df_bbox = rsna.get_rsna_bbox_dataframe(
                        rsna_path=rsna_root, image_size=image_size
                    )

                case "vindr":
                    _train_filenames, validate_filenames, test_filenames = (
                        _vindr_split_filenames(vindr_splits_dir, images_path)
                    )
                    df_bbox = vindr.get_vindr_bbox_dataframe(
                        vindr_root,
                        "all",
                        image_size=image_size,
                        # only the images this run loads: a test-only evaluation
                        # must not require every train image to be converted
                        image_ids={Path(name).stem for name in validate_filenames + test_filenames},
                        images_path=images_path,
                    )

                case _:
                    raise SystemExit(f"localization evaluation is not implemented for {dataset!r}")

            # create validation dataset
            validate_dataset = loader_utils.MultilabelDataset(
                df=df,
                images_path=images_path,
                img_shape=image_size,
                split="validate",
                split_pct1=0.8,
                split_pct2=1.0,
                possible_labels=possible_labels,
                filenames=validate_filenames,
                use_background_label=pred_background,
                df_bbox=df_bbox,
                multi_box_gt=dataset in TRANSFER_DATASETS,
                return_filename=return_filename,
                segmentation_path=segmentation_path,
                label_anatomical_regions=label_anatomical_regions,
                border_padding=border_padding,
            )
            validate_loader = th.utils.data.DataLoader(
                dataset=validate_dataset,
                batch_size=batch_size,
                num_workers=workers_evaluate,
                shuffle=False,
                collate_fn=loader_utils.custom_collate,
                worker_init_fn=loader_utils.dataloader_worker_init_fn,
            )

            # create test dataset
            test_dataset = loader_utils.MultilabelDataset(
                df=df,
                images_path=images_path,
                img_shape=image_size,
                split="test",
                split_pct1=0.8,
                split_pct2=1.0,
                possible_labels=possible_labels,
                filenames=test_filenames,
                use_background_label=pred_background,
                df_bbox=df_bbox,
                multi_box_gt=dataset in TRANSFER_DATASETS,
                return_filename=return_filename,
                segmentation_path=segmentation_path,
                label_anatomical_regions=label_anatomical_regions,
                border_padding=border_padding,
            )
            test_loader = th.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                num_workers=workers_evaluate,
                shuffle=False,
                collate_fn=loader_utils.custom_collate,
                worker_init_fn=loader_utils.dataloader_worker_init_fn,
            )

            evaluate_loader = [validate_loader, test_loader]

            # set None loss function
            criterion = None

        case _:
            raise SystemExit(
                f"unsupported evaluation split {split!r}; expected 'classify' or 'localize'"
            )

    # set save path of evaluation metrics in same dir as weights. Keep the
    # all-images localization output separate from the historical output.
    weights_path = Path(weights_info)

    ## Resolve the localization attention-layer count selection.
    # Precedence: explicit -alc/--attention_layer_count > a legacy equivalent
    # localization kwarg > the run's best-selection.json manifest (only for a
    # sweep run's model_best.pth) > the model/default n_attn_layers_avg.
    # The resolution validates the manifest strictly when it applies and
    # fails loudly on a malformed/inconsistent manifest.
    localization_selection = localization_selection_metadata(
        model=model,
        weights_path=weights_path,
        explicit_count=explicit_attention_layer_count,
        legacy_model_kwarg_count=legacy_n_attn_layers_avg,
    )
    if localization_selection["source"] != "default":
        manifest = localization_selection["manifest_path"]
        print(
            f"Localization attention-layer count selection: "
            f"{localization_selection['n_attn_layers_avg']} "
            f"(source: {localization_selection['source']})"
            + (f" [manifest: {manifest}]" if manifest is not None else "")
        )
    if localization_selection.get("override_outside_recorded_sweep"):
        print(
            "Note: the explicit attention-layer count is outside the "
            "recorded sweep counts of the run's best-selection.json"
        )
    if split == "localize" and localization_selection["source"] == "explicit_cli":
        # An explicit override must not overwrite the canonical
        # manifest-selected evaluation: append an unambiguous suffix.
        override_suffix = f"-attn-layers-{localization_selection['n_attn_layers_avg']}"
        if suffix and not suffix.startswith(override_suffix):
            override_suffix = f"{override_suffix}{suffix}"
        suffix = override_suffix

    localization_suffix = (
        "-all-images" if split == "localize" and not only_correctly_classified else ""
    )
    save_path = (
        weights_path.parent
        / f"{weights_path.stem}-evaluation-{split}{localization_suffix}{suffix}.pk"
    )

    if vindr_output_plan is not None:
        model = FindingOutputAdapter(model, vindr_output_plan)

    run_evaluation(
        device=device,
        parallel=parallel,
        jit_compile=jit_compile,
        model=model,
        criterion=criterion,
        evaluate_loader=evaluate_loader,
        save_path=save_path,
        split=split,
        mp_workers=mp_workers,
        iou_thr_merge=iou_thr_merge,
        proximity_thr=proximity_thr,
        localization_selection=localization_selection,
        only_correctly_classified=only_correctly_classified,
        use_amp=use_amp,
        bbox_method=bbox_method,
        selection=selection,
        optimize_metrics_separately=optimize_metrics_separately,
        fit_workers=fit_workers,
        frozen_calibration=frozen_calibration,
        **localization_kwargs,
    )
