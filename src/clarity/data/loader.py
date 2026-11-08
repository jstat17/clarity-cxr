from pathlib import Path
import numpy as np
import io
import cv2
import torch as th
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import pandas as pd
from tqdm import tqdm
from collections.abc import Callable
from typing import Any

from clarity.data import augment, files, labels as label_utils, segmentation as segm_utils, transform
from clarity.data.errors import InvalidDatasetError
from clarity.data.transform import warp_affine_channels

PYTORCH_SECOND_NORMALIZATION_DICT = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class MultilabelDataset(Dataset):
    """ChestX-ray14 multi-label dataset.

    Each image carries a multi-hot target over ``possible_labels`` (findings
    are not mutually exclusive), optionally with a leading background column
    (``use_background_label``). The dataset also serves the localization path:
    when ``df_bbox``/``label_anatomical_regions`` are supplied it returns the
    ground-truth boxes and per-label segmentation maps, and it can return the
    filename plus a precomputed teacher-label vector for distillation.

    Split selection is either explicit (``filenames``) or the historical
    patient-hash protocol (``split`` + ``split_pct1``/``split_pct2``). The
    deprecated alias ``MulticlassDataset`` points here; the name was misleading
    because the targets are multi-label, not multi-class.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_path: Path,
        img_shape: int | tuple[int, int],
        split: str,
        split_pct1: float,
        split_pct2: float,
        possible_labels: list[str],
        second_norm_dict: dict[str, list] | None = PYTORCH_SECOND_NORMALIZATION_DICT,
        filenames: list[str] | None = None,
        return_filename: bool = False,
        return_unaugmented: bool = False,
        augmentation_kwargs: dict | None = None,
        use_background_label: bool = False,
        load_images_in_memory: bool = False,
        precomputed_teacher_labels_path: Path | None = None,
        distil: bool = False,
        df_bbox: pd.DataFrame | None = None,
        multi_box_gt: bool = False,
        segmentation_path: Path | None = None,
        label_anatomical_regions: dict[str, list[str]] | None = None,
        segm_shape: int | tuple[int, int] | None = None,
        segm_dilate_kernel: int | None = None,
        segm_dilate_iterations: int | None = None,
        border_padding: int = 0,
    ) -> None:
        self.df = df.copy(deep=False)
        self.images_path = images_path
        self.precomputed_teacher_labels_path = precomputed_teacher_labels_path
        self.distil = distil
        self.segmentation_path = segmentation_path
        self.label_anatomical_regions = label_anatomical_regions

        if isinstance(img_shape, int):
            self.img_shape = (img_shape, img_shape)
        else:
            self.img_shape = img_shape

        self.split = split.casefold()
        self.possible_labels = possible_labels

        self.split_pct1 = split_pct1
        self.split_pct2 = split_pct2
        self.possible_hashes = 2**32
        self.hash_split1 = int(self.possible_hashes * self.split_pct1)
        self.hash_split2 = int(self.possible_hashes * self.split_pct2)

        if filenames is None:
            filenames = files.select_split_filenames(
                df=self.df,
                images_path=self.images_path,
                split=self.split,
                split_pct1=split_pct1,
                split_pct2=split_pct2,
            )
        self.filenames = files.add_file_extensions(filenames, self.images_path)

        self.use_background_label = use_background_label
        self.labels = th.from_numpy(self._get_labels())

        self.load_images_in_memory = load_images_in_memory

        self.second_norm_dict = second_norm_dict
        self.return_filename = return_filename

        self.augmentation_kwargs = augmentation_kwargs
        self.return_unaugmented = return_unaugmented

        if self.load_images_in_memory:
            self.compressed_images = self._load_all_images()

        # ``multi_box_gt`` keeps every box per (image, finding) instead of the
        # historical last-one-wins overwrite. Off by default: ChestX-ray14 has
        # at most one box per pair, and its published numbers depend on that
        # path being byte-identical. RSNA (up to 4 boxes/image) opts in.
        self.multi_box_gt = bool(multi_box_gt)
        if df_bbox is not None:
            self.image_abnorm_bboxes = self._get_bbox_dict(df_bbox)
        else:
            self.image_abnorm_bboxes = None

        if segm_shape is None:
            self.segm_shape = self.img_shape
        elif isinstance(segm_shape, int):
            self.segm_shape = (segm_shape, segm_shape)
        else:
            self.segm_shape = segm_shape

        if self.segmentation_path is not None:
            # the directory passed via -seg IS the input; classify it as the
            # npz cache or the legacy pk.xz source (no sibling-folder fallback)
            self.segm_cache_path, self.segmentation_read_path = (
                segm_utils.resolve_segmentation_paths(self.segmentation_path)
            )
            self.segmentation_filenames = files.add_file_extensions(
                self.filenames, self.segmentation_read_path
            )
            if self.segm_cache_path is not None and self.segmentation_filenames:
                # fail fast on format drift, shape mismatch, or a cache that
                # no longer matches the source pk.xz files
                segm_utils.validate_cache_meta(
                    cache_path=(
                        self.segmentation_read_path / self.segmentation_filenames[0]
                    ),
                    segm_shape=tuple(self.segm_shape),
                    segmentation_path=self.segmentation_path,
                    cache_filename=self.segmentation_filenames[0],
                    strict=True,
                )
            self.compressed_segmentations = (
                segm_utils.load_compressed_segmentations(
                    read_path=self.segmentation_read_path,
                    segmentation_filenames=self.segmentation_filenames,
                    split=self.split,
                    cached=self.segm_cache_path is not None,
                )
            )

        self.segm_dilate_kernel = segm_dilate_kernel
        self.segm_dilate_iterations = segm_dilate_iterations

        self.border_padding = border_padding

        self.epoch = -1
        self.teacher_labels = None

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch, and optionally load precomputed teacher labels.

        Args:
            epoch (int): Epoch number
        """
        self.epoch = epoch

        if self.precomputed_teacher_labels_path is not None:
            self._load_precomputed_teacher_labels()

    def _hash_value(self, val: Any) -> int:
        """Generate a 32-bit FNV-1a hash value for a given input.

        Args:
            val (Any): Value to hash

        Returns:
            int: 32-bit hash value
        """
        return transform.fnv1a_32(val)

    def _get_filenames(self) -> list[str]:
        """Filenames of the split, selected by the patient-hash protocol."""
        return files.select_split_filenames(
            df=self.df,
            images_path=self.images_path,
            split=self.split,
            split_pct1=self.split_pct1,
            split_pct2=self.split_pct2,
        )

    def _get_labels(self) -> np.ndarray:
        """Multi-hot label matrix for the selected filenames."""
        return label_utils.build_label_matrix(
            df=self.df,
            filenames=self.filenames,
            possible_labels=self.possible_labels,
            use_background_label=self.use_background_label,
            split=self.split,
        )

    def _load_precomputed_teacher_labels(self) -> None:
        """Load the precomputed teacher labels for the current epoch."""
        self.teacher_labels = label_utils.load_teacher_labels(
            teacher_labels_path=self.precomputed_teacher_labels_path,
            epoch=self.epoch,
            filenames=self.filenames,
            shape=tuple(self.labels.shape),
        )

    def _get_bbox_dict(
        self, df_bbox: pd.DataFrame
    ) -> dict[str, dict[int, dict[str, float] | list[dict[str, float]]]]:
        return label_utils.build_bbox_dict(
            df_bbox=df_bbox,
            possible_labels=self.possible_labels,
            use_background_label=self.use_background_label,
            multi_box=self.multi_box_gt,
        )

    def _unpack_cached_segmentation(
        self, cache_bytes: bytes
    ) -> dict[str, np.ndarray]:
        """Inflate a cached npz segmentation into a dict of boolean masks."""
        return segm_utils.unpack_cached_segmentation(cache_bytes, tuple(self.segm_shape))

    def _load_segmentation(self, idx: int, resize: bool = True) -> np.ndarray:
        """Load one instance's anatomical masks as per-label boolean masks."""
        segmentation = segm_utils.load_segmentation_dict(
            compressed=self.compressed_segmentations[idx],
            cached=self.segm_cache_path is not None,
            segm_shape=tuple(self.segm_shape),
            resize=resize,
        )
        segmentation = segm_utils.dilate_segmentation(
            segmentation,
            kernel_size=self.segm_dilate_kernel,
            iterations=self.segm_dilate_iterations,
        )
        return segm_utils.build_label_map(
            segmentation=segmentation,
            segm_shape=tuple(self.segm_shape),
            possible_labels=self.possible_labels,
            label_anatomical_regions=self.label_anatomical_regions,
            use_background_label=self.use_background_label,
        )

    def _structural_mask_build_ok(self) -> bool:
        """Whether the label map can use the memoized region-OR fast build.

        Requires the precomputed npz cache (masks already at segm_shape, so no
        per-sample resize), no dilation, and no border padding (the pad pass is
        then an identity and can be skipped entirely).
        """
        return (
            self.segmentation_path is not None
            and self.segm_cache_path is not None
            and not (self.segm_dilate_kernel and self.segm_dilate_iterations)
            and self.border_padding == 0
        )

    def _augment_masks_batched(
        self, segmentation: np.ndarray, rng_state, augmentations: list[Callable]
    ) -> np.ndarray:
        """Augment all label masks with one shared geometry per draw.

        Standard resets the RNG state before each mask, so every mask would
        sample the identical rotation angle and crop bounds; the transforms are
        therefore sampled once and applied to all masks together. Masks are
        stacked into one channel-last uint8 array and warped in batched OpenCV
        calls (chunks of at most 4 channels for warpAffine). Re-thresholding at
        >127 reproduces the boolean output of the per-mask path.
        """
        np.random.set_state(rng_state)
        transforms = augment.sample_similarity_transforms(augmentations)

        # group labels that share a region signature: they hold identical
        # source masks, so only the unique ones need warping.  The optional
        # background row is its own group (its mask is the complement of the
        # regions, so it cannot share a label's signature).
        signatures = []
        unique = {}
        group = []
        if getattr(self, "use_background_label", False):
            background_signature = ("__background__",)
            signatures.append(background_signature)
            unique[background_signature] = 0
            group.append(0)
        for label in self.possible_labels:
            signature = tuple(sorted(self.label_anatomical_regions[label]))
            signatures.append(signature)
            if signature not in unique:
                unique[signature] = len(unique)
            group.append(unique[signature])
        n_unique = len(unique)

        height, width = segmentation.shape[1:]
        unique_masks = [None] * n_unique
        for i, k in enumerate(group):
            if unique_masks[k] is None:
                unique_masks[k] = segmentation[i]

        stacked = np.stack(unique_masks, axis=-1).astype(np.uint8) * 255
        cur = stacked
        for kind, value in transforms:
            if kind == "rotate":
                matrix = cv2.getRotationMatrix2D(
                    ((width - 1) / 2.0, (height - 1) / 2.0), value, 1.0
                )
                cur = warp_affine_channels(cur, matrix)
            elif kind == "crop":
                cur = augment.batched_crop_resize(cur, value)

        out = cur > 127
        for i, k in enumerate(group):
            segmentation[i] = out[:, :, k]

        return segmentation

    def _load_all_images(self) -> list[bytes]:
        """Load all images into memory

        Returns:
            list[bytes]: List of compressed images as bytes
        """
        n = len(self)
        images = [None] * n

        # use tqdm progress bar
        pbar = tqdm(total=n, desc=f"Loading {self.split} images into memory", unit="image")

        for idx in range(n):
            image_path = self.images_path / self.filenames[idx]
            image = self._load_image(image_path)

            images[idx] = image
            pbar.update(1)

        # finish tqdm
        pbar.close()

        return images

    def _load_image(self, image_path: Path) -> np.ndarray | bytes:
        """Load image from image_path

        Args:
            image_path (Path): Path to the image on disk

        Returns:
            np.ndarray | bytes: Image, without converting type
        """
        # load image
        extension = image_path.suffix
        match extension:
            case ".png" | ".jpg" | ".jpeg":
                with open(image_path, "rb") as f:
                    image = f.read()
            case ".npz":
                image = np.load(image_path)["arr"]

        return image

    def _decompress_image(self, image: bytes | np.ndarray) -> np.ndarray:
        if isinstance(image, bytes):
            image = io.BytesIO(image)
            image = Image.open(image)
            image = np.array(image)

        return image

    def _normalize_image(
        self, image: np.ndarray, resize: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Normalize image for training

        Args:
            image (np.ndarray): Image as a numpy array
            resize (bool): Whether to resize the image. Defaults to True

        Returns:
            np.ndarray: 3-channel normalized image, shape (3, H, W)
        """
        # convert to float
        if np.issubdtype(image.dtype, np.integer):
            max_dtype_value = transform.get_max_value(image)
            image = image.astype(np.float32)

            # normalize to [0., 1.]
            image = image / max_dtype_value

        # ensure image is grayscale (single channel)
        if len(image.shape) == 3:
            image = image[:, :, 0]  # (H, W, C) -> (H, W)

        # resize
        if resize and image.shape != self.img_shape:
            image = transform.resize(image, self.img_shape)

        # copy to 3 channel dimensions
        image_3_channel = np.zeros(shape=(3, *image.shape), dtype=np.float32)
        for i in range(3):
            image_3_channel[i] = image

        image = image_3_channel  # (3, H, W)

        # do second normalization
        if self.second_norm_dict is not None:
            mean = self.second_norm_dict["mean"]
            mean = np.array(mean, dtype=np.float32)
            mean = mean[:, np.newaxis, np.newaxis]  # (3, 1, 1)

            std = self.second_norm_dict["std"]
            std = np.array(std, dtype=np.float32)
            std = std[:, np.newaxis, np.newaxis]  # (3, 1, 1)

            image = np.subtract(
                image,
                mean,
            )
            image = np.divide(image, std)

        return image

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, ...]:
        # set seed based on epoch and filename
        epoch = self.epoch
        filename = self.filenames[idx]

        val = f"{epoch}_{filename}"
        seed = self._hash_value(val)
        np.random.seed(seed)

        if not self.load_images_in_memory:
            # load image from disk
            image_path = self.images_path / filename
            image = self._load_image(image_path)

        else:
            # load image from memory
            image = self.compressed_images[idx]

        # decompress image
        image = self._decompress_image(image)
        # normalize and resize
        image = self._normalize_image(image, resize=True)
        # pad with border
        image = transform.border_pad_image(image, self.border_padding)

        # add augmentations
        if self.augmentation_kwargs is not None:
            image = image.copy()
            # create list of augmentations
            augmentations = augment.generate_augmentations_list(**self.augmentation_kwargs)
            # save internal RNG state for augmenting masks
            rng_state = np.random.get_state()
            # augment image
            image_aug = augment.apply_augmentations(image, augmentations)

            if self.return_unaugmented:
                image = (image, image_aug)
            else:
                image = image_aug

        output = []

        # convert to PyTorch tensor
        if isinstance(image, tuple):
            for img_norm in image:
                output.append(th.from_numpy(img_norm))
        else:
            output.append(th.from_numpy(image))

        # get labels from memory
        labels = self.labels[idx]
        output.append(labels)

        if self.return_filename:
            output.append(filename)

        # get teacher labels
        if self.teacher_labels is not None:
            teacher_labels = self.teacher_labels[idx]
            if self.distil:
                output[1] = teacher_labels.sigmoid()
            else:
                output.append(teacher_labels)

        # bounding boxes
        if self.image_abnorm_bboxes is not None:
            name, _extension = partition_filename(filename)
            # get this filename's bounding boxes
            # (have to use a custom collation function in wrapped DataLoader)
            abnorm_bboxes = self.image_abnorm_bboxes[name]
            output.append(abnorm_bboxes)

        # segmentation map
        if self.segmentation_path is not None:
            # label-map build: the memoized region-OR build only applies on the
            # npz cache without dilation/border padding; otherwise the exact
            # generic path runs (per-label OR over resized/dilated masks)
            segmentation = self._load_segmentation(idx, resize=True)
            if not self._structural_mask_build_ok():
                # the fast path skips this because border_padding == 0 there
                segmentation = segm_utils.pad_label_map(
                    segmentation, self.border_padding
                )

            # add augmentations
            if self.augmentation_kwargs is not None:
                if self._structural_mask_build_ok():
                    # batched geometry: masks receive only the similarity subset
                    # (rotate/crop) of the augmentation list, which is sampled
                    # once and applied to all masks in batched OpenCV calls
                    segmentation = self._augment_masks_batched(
                        segmentation, rng_state, augmentations
                    )
                else:
                    # loop over all label masks with per-mask RNG reset
                    for i in range(segmentation.shape[0]):
                        # reset internal RNG state to match image augmentations
                        np.random.set_state(rng_state)
                        # augment mask with only rotation and cropping
                        mask = transform.manipulate_boolean_mask(
                            mask=segmentation[i],
                            func=augment.apply_similarity_augmentations,
                            augmentations=augmentations,
                        )
                        segmentation[i] = mask

            output.append(segmentation)

        return tuple(output)


#: Deprecated alias: the class is a *multilabel* dataset (multi-hot targets),
#: not a multi-class one. Kept for external notebooks/scripts for one cycle.
MulticlassDataset = MultilabelDataset


class ClassAwareDistillationDataset(MultilabelDataset):
    """Multiclass dataset that also returns a precomputed teacher embedding.

    Used for distillation from precomputed teacher embeddings.  Embeddings are stored
    as ``<image stem>.npz`` files containing an ``embedding`` array; the file
    is looked up by stem (never by list position) so the image/embedding pairs
    cannot drift when the directory order changes.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_path: Path,
        img_shape: int | tuple[int, int],
        split: str,
        split_pct1: float,
        split_pct2: float,
        possible_labels: list[str],
        embeddings_path: Path,
        filenames: list[str] | None = None,
        second_norm_dict: dict[str, list] | None = PYTORCH_SECOND_NORMALIZATION_DICT,
        return_filename: bool = False,
    ) -> None:
        super().__init__(
            df=df,
            images_path=images_path,
            img_shape=img_shape,
            split=split,
            split_pct1=split_pct1,
            split_pct2=split_pct2,
            possible_labels=possible_labels,
            second_norm_dict=second_norm_dict,
            filenames=filenames,
            return_filename=return_filename,
        )
        self.embeddings_path = Path(embeddings_path)
        if not self.embeddings_path.is_dir():
            raise FileNotFoundError(f"embedding directory not found: {self.embeddings_path}")
        self.embedding_filenames = [
            self.embedding_filename_for(filename) for filename in self.filenames
        ]

    @staticmethod
    def embedding_filename_for(filename: str) -> str:
        """Return the ``.npz`` embedding filename for an image filename."""
        return f"{Path(filename).stem}.npz"

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_embedding(self, filename: str) -> np.ndarray:
        path = self.embeddings_path / self.embedding_filename_for(filename)
        if not path.is_file():
            raise FileNotFoundError(f"missing teacher embedding for {filename!r}: {path}")
        with np.load(path) as data:
            return data["embedding"]

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        image, cls_labels = item[0], item[1]
        filename = self.filenames[idx]
        embedding = th.from_numpy(np.asarray(self._load_embedding(filename), dtype=np.float32))
        if self.return_filename:
            return image, cls_labels, embedding, item[-1]
        return image, cls_labels, embedding


def custom_collate(batch):
    """
    Custom collate function for PyTorch DataLoader.

    This function handles batches where each item is a tuple returned by the dataset's __getitem__.
    For each position in the tuple:
    - If the elements are dictionaries, they are collected into a list without merging by key.
    - For all other types, PyTorch's default_collate is used for standard collation.
    If the batch items are dictionaries directly (not tuples), they are returned as a list.
    """
    # Handle empty batch
    if not batch:
        return batch

    first_item = batch[0]

    if isinstance(first_item, tuple):
        # Batch is a list of tuples; unzip into lists of elements at each position
        unzipped = list(zip(*batch))
        collated = []
        for elements in unzipped:
            # Check the type of the first element in this position (assuming type consistency)
            if isinstance(elements[0], dict):
                # Collect dictionaries into a list instead of merging by key
                collated.append(list(elements))
            else:
                # Use default_collate for tensors, numbers, etc.
                collated.append(default_collate(elements))
        # Return as a tuple to preserve the original structure
        return tuple(collated)

    elif isinstance(first_item, dict):
        # If batch items are dictionaries directly, return them as a list
        return list(batch)

    else:
        # For non-tuple, non-dict types, use default collation
        return default_collate(batch)


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Initializer for DataLoader worker processes.

    Each worker inherits the parent process's thread pools, so without this
    every OpenCV resize/warp would fan out to ~nproc threads and every worker
    would run its NumPy/torch CPU ops with the full core count — massively
    oversubscribing multi-worker setups on many-core machines (e.g. 20
    workers × 48 cores). Confine each worker to a single thread.
    """
    cv2.setNumThreads(0)
    # intra-op CPU threading inside workers (not exposed via DataLoader args)
    th.set_num_threads(1)


def _propagate_loader_options(loader_kwargs: dict) -> dict:
    """Return loader kwargs augmented with the repo-wide DataLoader options.

    Adds the shared ``worker_init_fn`` (see ``dataloader_worker_init_fn``)
    plus pin-memory / persistent-worker settings. Existing values set by the
    caller always take precedence.
    """
    kwargs = dict(loader_kwargs)
    kwargs.setdefault("worker_init_fn", dataloader_worker_init_fn)
    kwargs.setdefault("pin_memory", True)
    kwargs.setdefault("persistent_workers", kwargs.get("num_workers", 0) > 0)
    return kwargs


def update_dataloader(dataloader: DataLoader, epoch: int) -> DataLoader:
    """Update the epoch of a MultilabelDataset within a DataLoader object

    Args:
        dataloader (DataLoader): DataLoader object
        epoch (int): Epoch number

    Raises:
        InvalidDatasetError: DataLoader must use a MultilabelDataset dataset

    Returns:
        DataLoader: Updated DataLoader
    """
    # get dataloader params
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    num_workers = dataloader.num_workers
    sampler = dataloader.sampler
    drop_last = dataloader.drop_last

    # update epoch values
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if not isinstance(ds, MultilabelDataset):
                raise InvalidDatasetError(
                    f"Dataset in the DataLoader must be of type MultilabelDataset, not {type(ds)}"
                )

            ds.set_epoch(epoch)

    elif not isinstance(dataset, MultilabelDataset):
        raise InvalidDatasetError(
            f"Dataset in the DataLoader must be of type MultilabelDataset, not {type(dataset)}"
        )

    else:
        dataset.set_epoch(epoch)

    # re-create dataloader with updated dataset and existing params
    new_dataloader = DataLoader(
        **_propagate_loader_options(
            {
                "dataset": dataset,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "sampler": sampler,
                "drop_last": drop_last,
            }
        )
    )

    return new_dataloader


# Backwards-compatible names: the helpers now live in :mod:`data.files`.
add_file_extensions = files.add_file_extensions
get_file_extension_info = files.get_file_extension_info
partition_filename = files.partition_filename
