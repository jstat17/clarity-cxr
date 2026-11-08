"""External baseline arms of the transfer study (``clarity-baselines``).

Two published comparators are scored on the same split, label space and metric
as the CLARiTy arms, on RSNA Pneumonia or VinDr-CXR:

* ``txv``     -- TorchXRayVision DenseNet121-NIH, loaded unchanged (zero-shot);
* ``raddino`` -- frozen RAD-DINO behind a logistic-regression probe fitted on the
  target train list (C chosen on the validation list).

Usage::

    clarity-baselines classify --baseline txv     --dataset vindr -img DIR --splits-dir DIR -o txv.json
    clarity-baselines classify --baseline raddino --dataset vindr -img DIR --splits-dir DIR -o rd.json
    clarity-baselines localize --baseline raddino --dataset vindr -img DIR -seg DIR \\
        --splits-dir DIR --probes rd.probes.npz -o rd-loc.json

``localize`` reads the classifier through HiResCAM and fits the OneBox
calibration on the target validation split (Target-calibrated), then scores the
test split once. Needs ``uv sync --extra transfer``.
"""

from __future__ import annotations

import json
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from clarity.utils.cli import parse_path
from clarity.utils.paths import resolve_rsna_root, resolve_vindr_root
from clarity.external import runners


def _root(dataset: str, data_root: Path | None) -> Path:
    return resolve_rsna_root(data_root) if dataset == "rsna" else resolve_vindr_root(data_root)


def _save_probes(path: Path, probes: dict[str, tuple[np.ndarray, float]]) -> None:
    arrays = {}
    for index, (finding, (coef, intercept)) in enumerate(probes.items()):
        arrays[f"name_{index}"] = np.array(finding)
        arrays[f"coef_{index}"] = np.asarray(coef, dtype=np.float64)
        arrays[f"intercept_{index}"] = np.array(float(intercept))
    np.savez(path, **arrays)


def _load_probes(path: Path) -> dict[str, tuple[np.ndarray, float]]:
    payload = np.load(path, allow_pickle=False)
    count = sum(1 for key in payload.files if key.startswith("name_"))
    return {
        str(payload[f"name_{i}"]): (payload[f"coef_{i}"], float(payload[f"intercept_{i}"]))
        for i in range(count)
    }


def _json_safe(document: dict, drop: tuple[str, ...]) -> dict:
    return {key: value for key, value in document.items() if key not in drop}


def _common(parser: ArgumentParser) -> None:
    parser.add_argument("--baseline", required=True, choices=("txv", "raddino"))
    parser.add_argument("--dataset", required=True, choices=runners.DATASETS)
    parser.add_argument("-img", "--images", type=parse_path, required=True,
                        help="directory with the converted PNG images (TXV: 224 px; RAD-DINO: 512 px)")
    parser.add_argument("--splits-dir", type=parse_path, required=True,
                        help="directory with the three split lists (<dataset>_{train,val,test}_list.txt)")
    parser.add_argument("--data-root", type=parse_path, default=None,
                        help="dataset parent directory (overrides CLARITY_DATA_ROOT)")
    parser.add_argument("-o", "--output", type=parse_path, required=True, help="result JSON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke test on the first N val/test images (recorded in the JSON; not a result)")


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    classify = sub.add_parser("classify", help="classification AUROC on the val/test lists")
    _common(classify)
    classify.add_argument("--batch-size", type=int, default=32)
    classify.add_argument("--features-cache", type=parse_path, default=None,
                          help="RAD-DINO feature cache (.npz); reused when it exists")
    classify.add_argument("--probes-out", type=parse_path, default=None,
                          help="RAD-DINO probe weights for 'localize' (default: <output>.probes.npz)")

    localize = sub.add_parser("localize", help="HiResCAM + OneBox (Target-calibrated) localization")
    _common(localize)
    localize.add_argument("-seg", "--segmentations", type=parse_path, required=True,
                          help="anatomical segmentation directory")
    localize.add_argument("--probes", type=parse_path, default=None,
                          help="probe weights written by 'classify --baseline raddino'")
    localize.add_argument("--cam-layer", default="last",
                          help="TXV: norm5|denseblock3|denseblock2; RAD-DINO: block index (default last)")
    localize.add_argument("--batch-size", type=int, default=8)
    localize.add_argument("--workers", type=int, default=4)
    localize.add_argument("--fit-workers", type=int, default=1)

    args = parser.parse_args(argv)
    dataset = runners.get_dataset(args.dataset)
    root = _root(args.dataset, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "classify":
        if args.baseline == "txv":
            result = runners.classify_txv(
                dataset, root=root, splits_dir=args.splits_dir, images=args.images,
                device=args.device, batch_size=args.batch_size, limit=args.limit,
            )
        else:
            cache = args.features_cache or args.output.with_suffix(".features.npz")
            result = runners.classify_raddino(
                dataset, root=root, splits_dir=args.splits_dir, images=args.images,
                features_cache=cache, device=args.device, batch_size=args.batch_size,
                limit=args.limit,
            )
            probes_out = args.probes_out or args.output.with_suffix(".probes.npz")
            _save_probes(probes_out, result.pop("probes"))
            print(f"wrote {probes_out}")
        args.output.write_text(json.dumps(result, indent=1, default=float) + "\n")
        print(f"wrote {args.output}")
        return

    probes = _load_probes(args.probes) if args.probes is not None else None
    result = runners.localize(
        args.baseline, dataset, root=root, splits_dir=args.splits_dir, images=args.images,
        segmentations=args.segmentations, device=args.device, batch_size=args.batch_size,
        workers=args.workers, fit_workers=args.fit_workers, limit=args.limit,
        cam_layer=args.cam_layer, probes=probes,
    )
    pickle_path = args.output.with_suffix(".pk")
    with open(pickle_path, "wb") as handle:
        pickle.dump(
            {"test_instances": result["test_instances"], "test_filenames": result["test_filenames"],
             "localization_method": result["localization_method"]},
            handle,
        )
    args.output.write_text(
        json.dumps(_json_safe(result, ("test_instances", "test_filenames")), indent=1, default=float) + "\n"
    )
    print(f"wrote {args.output} and {pickle_path}")


if __name__ == "__main__":
    main()
