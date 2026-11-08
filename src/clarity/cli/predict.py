import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle as pk
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from clarity.data import chestxray14
from clarity.data.loader import MultilabelDataset, update_dataloader
from clarity.utils.cli import (
    parse_kwargs_into_dict,
    parse_truthy,
    parse_casefold,
    parse_path,
)
from clarity.utils.runtime import get_autocast_context, set_primary_device, wrap_parallel, parse_weights
from clarity.models.factory import get_model
from clarity.utils.cli import add_standard_aliases
from clarity.utils.paths import resolve_cxr14_root
from clarity.training.setup import AUGMENTATION_KWARGS

"""Save per-epoch classification predictions for a trained model.

Usage::

    clarity-predict [model_name] [model_size] <weights> [options]
"""


def run_predict(
    device: str,
    parallel: bool,
    jit_compile: bool,
    model: nn.Module,
    evaluate_loader: DataLoader,
    num_epochs: int,
    start_epoch: int,
    save_dir: Path,
    force: bool,
    use_amp: bool = False,
) -> None:
    # use DataParallel if parallel is enabled
    model, device = wrap_parallel(model, parallel, device)

    # Set the current CUDA device when applicable. CPU execution must not
    # call torch.cuda.set_device().
    device = set_primary_device(device)

    # send model to device
    model = model.to(device)

    # compile model
    if jit_compile:
        model.compile()

    model.eval()
    for epoch in range(start_epoch, num_epochs):
        # set save path
        epoch_logits_filename = f"epoch_{epoch:03}.pk"
        epoch_logits_save_dir = save_dir / epoch_logits_filename

        # skip to next epoch if results exist on disk
        if not force and epoch_logits_save_dir.exists():
            print(f"Skipping epoch {epoch}, results exist")
            continue

        # set epoch for dataset
        evaluate_loader = update_dataloader(evaluate_loader, epoch)

        # create pbar
        pbar = tqdm(iterable=evaluate_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        epoch_logits_dict = dict()

        # evaluate all batches
        with th.no_grad():
            for batch in pbar:
                images, _labels, filenames = batch
                images = images.to(device, non_blocking=True)

                # get classifier predictions as logits
                with get_autocast_context(device, enabled=use_amp):
                    logits = model(images)
                logits = logits.detach().float().cpu().numpy()

                # save to dictionary
                for logits_vec, filename in zip(logits, filenames):
                    epoch_logits_dict[filename] = logits_vec

        # save results to disk
        with open(epoch_logits_save_dir, "wb") as f:
            pk.dump(epoch_logits_dict, f)


def main() -> None:
    # set up argument parsing
    parser = ArgumentParser(
        description="Save the predictions of a trained model on the given dataset."
    )
    parser.add_argument("model_name", type=parse_casefold, help="Trained model to load")
    parser.add_argument("model_size", type=parse_casefold, help="Model size")
    parser.add_argument("weights", type=parse_weights, help="Model weights")
    parser.add_argument(
        "-img", "--images_path", type=parse_path, default=None, help="Path to images"
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=parse_casefold,
        default="chestxray14",
        help="Dataset of the images (default 'chestxray14')",
    )
    parser.add_argument("-s", "--size", type=int, default=224, help="Image size (default 224)")
    parser.add_argument(
        "-d",
        "--device",
        type=parse_casefold,
        default="cuda:0",
        help="The device used to evaluate the model ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "-amp",
        "--use_amp",
        type=parse_truthy,
        default=False,
        help="Use bfloat16 autocast for model forward passes (default False)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=parse_truthy,
        default=True,
        help="Whether to evaluate in parallel on multiple GPUs",
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
        help="Number of epochs to evaluate for (default 100)",
    )
    parser.add_argument(
        "-sn", "--start_epoch", type=int, default=0, help="The starting epoch number (default 0)"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size (default 64)")
    parser.add_argument(
        "-we",
        "--workers_evaluate",
        type=int,
        default=8,
        help="Number of evaluation dataloader workers (default 8)",
    )
    parser.add_argument(
        "-me",
        "--load_memory_evaluate",
        type=parse_truthy,
        default=False,
        help="Whether to load all evaluation images into memory (default False)",
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
        help="Pad images with a border of this size (default 0)",
    )
    parser.add_argument(
        "-mkw",
        "--model_kwargs",
        type=parse_kwargs_into_dict,
        default={},
        help="Model keyword arguments as 'key1=value1;key2=value2'. Values are parsed as Python literals (default {})",
    )
    parser.add_argument(
        "-f",
        "--force",
        type=parse_truthy,
        default=False,
        help="Force save predictions for an epoch even if it has been saved prior (default False)",
    )
    parser.add_argument(
        "--data-root",
        type=parse_path,
        default=None,
        help="Dataset root (overrides CLARITY_DATA_ROOT)",
    )
    add_standard_aliases(parser)

    # parse command line arguments
    args = parser.parse_args()
    cxr14_root = resolve_cxr14_root(args.data_root)
    if args.images_path is None:
        args.images_path = cxr14_root

    model_name = args.model_name
    model_size = args.model_size
    weights, weights_info = args.weights

    images_path = args.images_path
    dataset = args.dataset
    image_size = args.size

    device = args.device
    parallel = args.parallel
    jit_compile = args.jit_compile
    use_amp = args.use_amp

    num_epochs = args.num_epochs
    start_epoch = args.start_epoch
    batch_size = args.batch_size

    workers_evaluate = args.workers_evaluate
    load_memory_evaluate = args.load_memory_evaluate

    pred_background = args.pred_background
    border_padding = args.border_padding
    model_kwargs = args.model_kwargs

    force = args.force

    # get dataset info
    match dataset:
        case "chestxray14":
            df = chestxray14.get_chestxray14_dataframe(cxr14_root)
        case _:
            raise SystemExit(f"unsupported public dataset {dataset!r}; only chestxray14 is shipped")

    possible_labels = chestxray14.CXR_14_ABNORMALITIES
    num_classes = len(possible_labels)

    # load model based on arguments
    model = get_model(
        model_name=model_name,
        model_size=model_size,
        num_classes=num_classes,
        weights=weights,
        pred_background=pred_background,
        **model_kwargs,
    )

    # dataset and dataloader
    evaluate_dataset = MultilabelDataset(
        df=df,
        images_path=images_path,
        img_shape=image_size,
        split="train",
        split_pct1=1.0,
        split_pct2=1.0,
        possible_labels=possible_labels,
        augmentation_kwargs=AUGMENTATION_KWARGS,
        use_background_label=pred_background,
        load_images_in_memory=load_memory_evaluate,
        return_filename=True,
        border_padding=border_padding,
    )

    evaluate_loader = DataLoader(
        dataset=evaluate_dataset,
        batch_size=batch_size,
        num_workers=workers_evaluate,
        shuffle=False,
        drop_last=False,
    )

    # create save folder
    weights_path = Path(weights_info)
    predictions_save_folder = f"{weights_path.stem}-{dataset}-predictions"
    predictions_save_dir = weights_path.parent / predictions_save_folder
    os.makedirs(predictions_save_dir, exist_ok=True)

    run_predict(
        device=device,
        parallel=parallel,
        jit_compile=jit_compile,
        model=model,
        evaluate_loader=evaluate_loader,
        num_epochs=num_epochs,
        start_epoch=start_epoch,
        save_dir=predictions_save_dir,
        force=force,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()
