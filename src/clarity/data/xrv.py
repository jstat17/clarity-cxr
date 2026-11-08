"""TorchXRayVision-format dataset (optional ``[data]`` extra).

Kept separate from :mod:`data.loader` so importing the main dataset never
requires ``torchxrayvision``.  Only ``clarity-segment`` uses this class.
"""

from __future__ import annotations

import logging

import os
from pathlib import Path

import numpy as np
import skimage
import torch as th
from torch.utils.data import Dataset

from clarity.data import transform
from clarity.data.errors import InvalidDatasetError


logger = logging.getLogger(__name__)
#: image extensions this dataset accepts; anything else in the directory
#: (sidecar metadata, checksums, DICOM sources) is not an image
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npz"})


class XRVDataset(Dataset):
    """Dataset for TorchXRayVision models, which loads images from a path, resizes them to 512 x 512,\
    normalizes to the [-1024, 1024] range as float32

    Only files with a known image extension are enumerated: an image directory
    may legitimately carry sidecar files (e.g. the DICOM->PNG conversion's
    ``_conversion_meta.json``), and submitting one to ``skimage.io.imread``
    raises an ``OSError`` that the loader turns into a ``None`` sample, which
    then crashes ``default_collate`` far from the cause.
    """

    set_shape: tuple[int, int] = (512, 512)

    def __init__(self, images_path: Path, save_path_images: Path | None = None) -> None:
        from clarity.data.loader import add_file_extensions, get_file_extension_info

        self.images_path = images_path
        self.filenames = [
            filename
            for filename in os.listdir(self.images_path)
            if Path(filename).suffix.casefold() in IMAGE_EXTENSIONS
        ]
        if not self.filenames:
            raise ValueError(
                f"no image files found in {self.images_path} (accepted extensions: "
                f"{sorted(IMAGE_EXTENSIONS)})"
            )

        # remove images that have already been segmented previously
        if save_path_images is not None:
            filenames_existing, _ = get_file_extension_info(os.listdir(save_path_images))
            filenames, _ = get_file_extension_info(self.filenames)

            filenames = sorted(set(filenames) - set(filenames_existing))
            self.filenames = add_file_extensions(filenames, self.images_path)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        try:
            # load image
            image_path = self.images_path / Path(self.filenames[idx])
            image = skimage.io.imread(image_path)

            # ensure image is grayscale (single channel)
            if len(image.shape) == 3:
                image = image[:, :, 0]  # (H, W, C) -> (H, W)

            original_size = list(image.shape)  # (height, width)

            # resize to 512 x 512
            image = transform.resize(image, self.set_shape)
            image = image[np.newaxis, :]  # add channel dimension (1, H, W)

            # normalize to [-1024, 1024] and convert to float
            import torchxrayvision as xrv

            max_dtype_value = transform.get_max_value(image)
            image = xrv.datasets.normalize(image, max_dtype_value)

            # convert to PyTorch tensor
            tensor = th.from_numpy(image).float()

            return tensor, th.tensor(original_size), self.filenames[idx]

        except OSError as e:
            raise InvalidDatasetError(f"failed to read image {image_path}: {e}") from e
