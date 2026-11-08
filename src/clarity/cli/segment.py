from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import pickle as pk
import lzma
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor, wait

from clarity.data.xrv import XRVDataset
from clarity.utils.cli import parse_casefold, parse_path, parse_truthy
from clarity.utils.runtime import (
    get_autocast_context,
    set_primary_device,
    wrap_parallel,
)

"""Segment anatomical regions in a chest X-ray with the ChestX-Det model.

Usage::

    clarity-segment /path/to/images /path/to/segmentations [options]
"""


def save_segmentation_maps(segmentations: np.ndarray, labels: list[str], save_path: Path) -> None:
    # create dictionary from clarity.label to region mask
    d = dict()
    for label, segmentation in zip(labels, segmentations):
        d[label] = segmentation

    # save to xz-compressed pickle file
    with lzma.open(save_path, "wb") as f:
        pk.dump(d, f)


def run_segment(
    images_path: Path,
    save_dir: Path,
    batch_size: int,
    num_workers: int,
    mp_workers: int,
    device: str,
    parallel: bool,
    force: bool,
    use_amp: bool = False,
):
    # create save dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # prepare dataset and dataloader
    if force:
        dataset = XRVDataset(images_path)
    else:
        dataset = XRVDataset(images_path, save_dir)

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # Optional dependency: keep it lazy so ``--help`` works without the
    # ``data`` extra installed.
    try:
        import torchxrayvision as xrv
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "clarity-segment requires the optional 'data' extra "
            "(torchxrayvision); install it with `uv sync --extra data`"
        ) from exc

    # load segmentation model
    segm_model = xrv.baseline_models.chestx_det.PSPNet()
    target_labels = segm_model.targets

    # use DataParallel if parallel is enabled
    segm_model, device = wrap_parallel(segm_model, parallel, device)

    # Set the current CUDA device when applicable. CPU execution must not
    # call torch.cuda.set_device().
    device = set_primary_device(device)

    # send model to device
    segm_model = segm_model.to(device)

    # set model to evaluation mode
    segm_model.eval()

    # create progess bar
    pbar = tqdm(dataloader, desc="Segmenting anatomical regions", unit="batch")

    # store future objects
    futures = []

    # create thread pool for saving segmentation maps
    with ProcessPoolExecutor(max_workers=mp_workers) as exec:
        # loop over each batch
        with th.no_grad():
            for batch in pbar:
                # get data
                tensor_batch, _original_sizes, filenames = batch
                tensor_batch = tensor_batch.to(device, non_blocking=True)

                # get predictions
                with get_autocast_context(device, enabled=use_amp):
                    pred_batch = segm_model(tensor_batch)

                # Keep inference post-processing in float32 after the
                # autocast model forward.
                pred_batch = pred_batch.float()
                pred_batch = 1 / (1 + th.exp(-pred_batch))
                pred_segm = pred_batch > 0.5

                # move to cpu and numpy
                pred_segm = pred_segm.float().cpu().numpy()

                # loop over each image in batch
                for segmentations, filename in zip(pred_segm, filenames):
                    # create save path
                    segm_filename = filename.partition(".")[0] + ".pk.xz"
                    segm_save_path = save_dir / segm_filename

                    # save masks to disk
                    future = exec.submit(
                        save_segmentation_maps,
                        segmentations=segmentations,
                        labels=target_labels,
                        save_path=segm_save_path,
                    )
                    futures.append(future)

    # wait until all segmentation maps saved
    wait(futures)


def main() -> None:
    # set up argument parsing
    parser = ArgumentParser(
        description="Segment anatomical regions in a Chest X-ray using the ChestX-Det Segmentation Model."
    )
    parser.add_argument("images_path", type=parse_path, help="Path to chest X-ray images")
    parser.add_argument("save_dir", type=parse_path, help="Desired save directory for masks")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size of images to load at once (default 32)",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=1,
        help="Number of CPU workers to use for the data loader (default 1)",
    )
    parser.add_argument(
        "-mpw",
        "--mp_workers",
        type=int,
        default=1,
        help="Number of CPU workers to use for the process pool (default 1)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=parse_casefold,
        default="cuda:0",
        help="Devce to load the segmentation model and data onto (default 'cuda:0')",
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
        help="Split computation across multiple GPUs in parallel (default True)",
    )
    parser.add_argument(
        "-f",
        "--force",
        type=parse_truthy,
        default=False,
        help="Force segment even if images have already been segmented previously (default False)",
    )

    # parse command line arguments
    args = parser.parse_args()

    images_path = args.images_path
    save_dir = args.save_dir

    batch_size = args.batch_size
    num_workers = args.num_workers
    mp_workers = args.mp_workers

    device = args.device
    parallel = args.parallel
    force = args.force
    use_amp = args.use_amp

    run_segment(
        images_path=images_path,
        save_dir=save_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mp_workers=mp_workers,
        device=device,
        parallel=parallel,
        force=force,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()
