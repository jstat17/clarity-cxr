"""Training setup: dataset, model, optimizer, loss, loader and run artifacts.

The CLI module :mod:`clarity.cli.train` only parses arguments and delegates here.
Kept importable without side effects so tests can patch source modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch as th
import torch.optim as optim

from clarity.data import chestxray14
from clarity.data import splits as split_manifest
from clarity.losses.registry import get_loss
from clarity.training.runner import run_training_loop
from clarity.training.sweep_spec import build_localization_sweep_spec
from clarity.data import loader as loader_utils
from clarity.training.scheduler import get_cosine_schedule_with_warmup
from clarity.utils.io import get_terminal_command
from clarity.training.loop import get_next_run_folder, set_global_seed
from clarity.models import factory as vision_get

AUGMENTATION_KWARGS = {
    "rotate_prob": 1.0,
    "crop_prob": 1.0,
    "brightness_prob": 1.0,
    "noise_prob": 1.0,
    "n_noise": 3,
    "rotate_angle_min": -15.0,
    "rotate_angle_max": 15.0,
    "crop_frac_min": 0.7,
    "crop_frac_max": 1.0,
    "brightness_min": -0.25,
    "brightness_max": 0.25,
    "poisson_frac_min": 0.0,
    "poisson_frac_max": 1.0,
}


def run_training(
    args,
    *,
    cxr14_root: Path,
    models_root: Path,
    split_manifest_data,
) -> None:
    """Build the run from parsed args and execute the training loop."""
    set_global_seed(args.seed if args.seed is not None else 0)
    model_name = args.model_name
    model_size = args.model_size

    images_path = args.images_path[0]
    segmentation_path = args.segmentation_path[0]
    dataset = args.dataset[0]
    precomputed_teacher_labels_path = args.precomputed_teacher_labels_path[0]

    image_size = args.size
    device = args.device
    parallel = args.parallel
    jit_compile = args.jit_compile
    use_amp = args.use_amp

    num_epochs = args.num_epochs
    early_stop_num_epochs = args.early_stop_num_epochs
    terminate_epoch = args.terminate_epoch

    batch_size = args.batch_size
    warmup_steps = args.warmup_steps
    accumulation_factor = args.accumulation_factor

    optimizer_name = args.optimizer
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    loss = args.loss

    split_pct1 = args.split_pct1
    split_pct2 = args.split_pct2
    split = args.split

    workers_train = args.workers_train
    workers_validate = args.workers_validate
    mp_workers = args.mp_workers

    load_memory_train = args.load_memory_train
    load_memory_validate = args.load_memory_validate

    weights, weights_info = args.weights
    model_kwargs = args.model_kwargs
    loss_kwargs = args.loss_kwargs

    augment = args.augment
    if augment:
        augmentation_kwargs = {
            **AUGMENTATION_KWARGS,
            **dict(getattr(args, "augmentation_kwargs", None) or {}),
        }
    else:
        augmentation_kwargs = None

    pred_background = args.pred_background
    border_padding = args.border_padding

    train_localization = args.train_localization
    if args.segm_size is None:
        # default: keep segmentation maps at the model's input image resolution
        segm_size = (image_size, image_size)
    else:
        segm_size = (args.segm_size, args.segm_size)
    segm_dilate_kernel = args.segm_dilate_kernel
    segm_dilate_iterations = args.segm_dilate_iterations

    validate_localization = args.validate_localization
    localization_kwargs = dict(args.localization_kwargs)
    # Localization metrics include incorrectly classified abnormalities by default.
    # Keep the historical filtered behavior available through -lckw if needed.
    localization_kwargs.setdefault("only_correctly_classified", False)
    # Explicit method flags take precedence over any -lckw duplicate so the
    # method is visible in the source/command, not inherited silently.
    localization_kwargs.pop("bbox_method", None)
    localization_kwargs.pop("selection", None)
    localization_kwargs["bbox_method"] = args.bbox_method
    if args.selection_strategy is not None:
        localization_kwargs["selection"] = args.selection_strategy
    localization_kwargs["optimize_metrics_separately"] = bool(args.localization_optimize_separately)
    localization_kwargs["fit_workers"] = args.localization_fit_workers

    info = args.info

    # localization attention-layer sweep (opt-in)
    sweep_counts = args.localization_attention_layer_sweep
    sweep_primary = args.localization_attention_primary
    if sweep_counts is None and sweep_primary is not None:
        raise SystemExit(
            "-lasp/--localization_attention_primary is only allowed together "
            "with -las/--localization_attention_layer_sweep; no sweep is "
            "enabled by default"
        )

    # get dataset info
    match dataset:
        case "chestxray14":
            df = chestxray14.get_chestxray14_dataframe(cxr14_root)
            possible_labels = chestxray14.CXR_14_ABNORMALITIES
            label_anatomical_regions = chestxray14.CXR_14_ABNORMALITY_REGIONS
        case _:
            raise SystemExit(f"unsupported public dataset {dataset!r}; only chestxray14 is shipped")

    num_classes = len(possible_labels)

    # load model based on arguments
    model = vision_get.get_model(
        model_name=model_name,
        model_size=model_size,
        num_classes=num_classes,
        weights=weights,
        pred_background=pred_background,
        **model_kwargs,
    )

    # set optimizer
    match optimizer_name:
        case "adamw":
            optimizer = optim.AdamW(
                params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        case "adam":
            optimizer = optim.Adam(
                params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        case _:
            raise SystemExit(
                f"unsupported optimizer {optimizer_name!r}; expected 'adamw' or 'adam'"
            )

    # select dataset split
    match split.partition("-")[0]:
        # use a random split dictated by split_pct1 and split_pct2 as train-val
        case "random":
            train_filenames = None
            validate_filenames = None

        # use the official train-test split as train-val (80-20)
        case "official":
            match dataset:
                case "chestxray14":
                    train_filenames, validate_filenames = chestxray14.get_official_split_filenames(
                        chestxray14_path=cxr14_root, available_images_path=images_path
                    )
                    split_pct1 = 0.8
                    split_pct2 = 1.0

        # use stratified sampling of official train as train-val (70-10)
        case "classify":
            train_size = int(split.split("-", 1)[1]) if "-" in split else 70
            if train_size not in (70, 75):
                raise SystemExit(
                    f"unsupported classification split {split!r}; expected "
                    "'classify', 'classify-70' or 'classify-75'"
                )
            match dataset:
                case "chestxray14":
                    if split_manifest_data is not None:
                        train_filenames = split_manifest.filter_filenames(
                            split_manifest_data.classification_train,
                            images_path,
                            chestxray14.IGNORED_IMAGES,
                        )
                        validate_filenames = split_manifest.filter_filenames(
                            split_manifest_data.classification_val,
                            images_path,
                            chestxray14.IGNORED_IMAGES,
                        )
                    else:
                        train_filenames, validate_filenames = (
                            chestxray14.get_classification_split_filenames(
                                chestxray14_path=cxr14_root,
                                available_images_path=images_path,
                                train_size=train_size,
                            )
                        )
                    split_pct1 = train_size / 100
                    split_pct2 = 0.8

        case _:
            raise SystemExit(
                f"unsupported training split {split!r}; expected 'random', "
                "'official' or 'classify-<70|75>'"
            )

    # set up localization dataloader for in-training validation
    if validate_localization:
        match dataset:
            case "chestxray14":
                if split_manifest_data is not None:
                    localize_val_filenames = split_manifest.filter_filenames(
                        split_manifest_data.localization_val,
                        images_path,
                        chestxray14.IGNORED_IMAGES,
                    )
                else:
                    localize_val_filenames, _localize_test_filenames = (
                        chestxray14.get_localization_split_filenames(chestxray14_path=cxr14_root)
                    )
                df_bbox = chestxray14.get_chestxray14_bbox_dataframe(
                    chestxray14_path=cxr14_root, image_size=image_size
                )

        validate_localization_dataset = loader_utils.MultilabelDataset(
            df=df,
            images_path=images_path,
            img_shape=image_size,
            split="validate",
            split_pct1=0.5,
            split_pct2=0.5,
            possible_labels=possible_labels,
            filenames=localize_val_filenames,
            use_background_label=pred_background,
            df_bbox=df_bbox,
            return_filename=True,
            load_images_in_memory=True,
            segmentation_path=segmentation_path,
            label_anatomical_regions=label_anatomical_regions,
            border_padding=border_padding,
        )
        validate_localization_loader = th.utils.data.DataLoader(
            **loader_utils._propagate_loader_options(
                {
                    "dataset": validate_localization_dataset,
                    "batch_size": batch_size,
                    "num_workers": workers_validate,
                    "shuffle": False,
                    "collate_fn": loader_utils.custom_collate,
                }
            )
        )

    else:
        validate_localization_loader = None

    # datasets, loaders and samplers
    train_dataset = loader_utils.MultilabelDataset(
        df=df,
        images_path=images_path,
        img_shape=image_size,
        split="train",
        split_pct1=split_pct1,
        split_pct2=split_pct2,
        possible_labels=possible_labels,
        filenames=train_filenames,
        augmentation_kwargs=augmentation_kwargs,
        use_background_label=pred_background,
        load_images_in_memory=load_memory_train,
        segmentation_path=segmentation_path if train_localization else None,
        label_anatomical_regions=label_anatomical_regions,
        segm_shape=segm_size,
        segm_dilate_kernel=segm_dilate_kernel,
        segm_dilate_iterations=segm_dilate_iterations,
        precomputed_teacher_labels_path=precomputed_teacher_labels_path,
        distil=(precomputed_teacher_labels_path is not None),
        border_padding=border_padding,
    )

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": batch_size,
        "num_workers": workers_train,
        "shuffle": True,
        "drop_last": False,
    }

    train_loader = th.utils.data.DataLoader(
        **loader_utils._propagate_loader_options(train_loader_kwargs)
    )

    validate_dataset = loader_utils.MultilabelDataset(
        df=df,
        images_path=images_path,
        img_shape=image_size,
        split="validate",
        split_pct1=split_pct1,
        split_pct2=split_pct2,
        possible_labels=possible_labels,
        filenames=validate_filenames,
        use_background_label=pred_background,
        load_images_in_memory=load_memory_validate,
        segmentation_path=segmentation_path if train_localization else None,
        label_anatomical_regions=label_anatomical_regions,
        border_padding=border_padding,
    )
    validate_loader = th.utils.data.DataLoader(
        **loader_utils._propagate_loader_options(
            {
                "dataset": validate_dataset,
                "batch_size": batch_size,
                "num_workers": workers_validate,
                "shuffle": False,
                "drop_last": False,
            }
        )
    )

    # set loss function
    criteria = []
    if not isinstance(loss, list):
        loss = [loss]

    # get dataset labels
    labels = train_dataset.labels

    for loss_str in loss:
        criteria.append(
            get_loss(
                loss_name=loss_str,
                pred_background=pred_background,
                labels=labels,
                device=device,
                **loss_kwargs,
            )
        )

    # define learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_epochs
    )

    # build and validate the localization sweep specification *before* any
    # run folder is created, so a rejected command leaves no empty run_XXX
    localization_sweep = None
    if sweep_counts is not None:
        localization_sweep = build_localization_sweep_spec(
            model=model,
            criteria=criteria,
            counts=sweep_counts,
            primary=sweep_primary,
            validate_localization=validate_localization,
            localization_kwargs=localization_kwargs,
        )

    # creating save directory
    save_dir_root = models_root / model.fullname
    os.makedirs(save_dir_root, exist_ok=True)

    # use the explicitly requested run folder when given (an external job runner can
    # use this to pre-assign static folders and avoid races)
    if args.run_folder is not None:
        save_dir = (
            args.run_folder if args.run_folder.is_absolute() else save_dir_root / args.run_folder
        )
    else:
        save_dir = get_next_run_folder(save_dir_root)

    os.makedirs(save_dir, exist_ok=True)

    # save hyperparams log
    hyperparams = {
        "model_name": model_name,
        "model_size": model_size,
        "num_classes": num_classes,
        "images_path": [str(pth) for pth in args.images_path],
        "segmentation_path": [str(pth) for pth in args.segmentation_path],
        "dataset": args.dataset,
        "image_size": image_size,
        "device": device,
        "parallel": parallel,
        "jit_compile": jit_compile,
        "use_amp": use_amp,
        "num_epochs": num_epochs,
        "early_stop_num_epochs": (
            None if early_stop_num_epochs == float("inf") else early_stop_num_epochs
        ),
        "terminate_epoch": (None if terminate_epoch == float("inf") else terminate_epoch),
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "accumulation_factor": accumulation_factor,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_name": loss,
        "criterion": [criterion.dict_repr() for criterion in criteria],
        "scheduler": scheduler.dict_repr(),
        "split_pct1": split_pct1,
        "split_pct2": split_pct2,
        "split": split,
        "workers_train": workers_train,
        "workers_validate": workers_validate,
        "mp_workers": mp_workers,
        "weights": weights_info,
        "model_kwargs": model_kwargs,
        "loss_kwargs": loss_kwargs,
        "augmentation_kwargs": augmentation_kwargs,
        "seed": args.seed if args.seed is not None else 0,
        "pred_background": pred_background,
        "train_localization": train_localization,
        "border_padding": border_padding,
        "segm_shape": segm_size,
        "segm_dilate_kernel": segm_dilate_kernel,
        "segm_dilate_iterations": segm_dilate_iterations,
        "validate_localization": validate_localization,
        "localization_kwargs": localization_kwargs,
        "precomputed_teacher_labels_path": str(precomputed_teacher_labels_path),
        "info": info,
        "command": get_terminal_command(),
        "localization_sweep": (
            {
                "attention_layer_counts": list(localization_sweep.attention_layer_counts),
                "primary_attention_layer_count": localization_sweep.primary_attention_layer_count,
                "selection_metric": localization_sweep.selection_metric,
                "early_stopping_policy": localization_sweep.early_stopping_policy,
                "exact": localization_sweep.exact,
            }
            if localization_sweep is not None
            else None
        ),
    }

    hyperparams_log_path = save_dir / "hyperparams.json"
    with open(hyperparams_log_path, "w") as f:
        json.dump(hyperparams, f, indent=2)

    run_training_loop(
        device=device,
        parallel=parallel,
        jit_compile=jit_compile,
        model=model,
        criterion=criteria,
        optimizer=optimizer,
        num_epochs=num_epochs,
        early_stop_num_epochs=early_stop_num_epochs,
        terminate_epoch=terminate_epoch,
        train_loader=train_loader,
        evaluate_loader=validate_loader,
        accumulation_factor=accumulation_factor,
        scheduler=scheduler,
        save_dir=save_dir,
        validate_localization_loader=validate_localization_loader,
        mp_workers=mp_workers,
        localization_kwargs=localization_kwargs,
        use_amp=use_amp,
        localization_sweep=localization_sweep,
    )
