"""TXV input contract: one channel, ``[-1024, 1024]``, the checkpoint's resolution.

The resize goes through the library's own ``XRayResizer`` on purpose. Re-implementing
that transform is how an external comparison silently stops being "their model".
"""

from __future__ import annotations

import numpy as np
import torchxrayvision as xrv


class XRVPreprocessor:
    """Turn a uint8 grayscale array into the tensor TXV checkpoints expect."""

    def __init__(self, resolution: int) -> None:
        self.resolution = int(resolution)
        # instantiated here (not per image) so the resize engine is created once
        self._resize = xrv.datasets.XRayResizer(self.resolution)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 3:
            image = image[:, :, 0]  # same collapse the study's loader uses
        if image.ndim != 2:
            raise ValueError(f"expected a 2-D grayscale image, got {image.shape}")

        if np.issubdtype(image.dtype, np.integer):
            max_value = float(np.iinfo(image.dtype).max)
        else:
            max_value = float(np.max(np.abs(image))) or 1.0
        # `reshape=True` is not cosmetic: `XRayResizer` is called on a `(C, H, W)`
        # array by every library dataset. Handing it a 2-D `(H, W)` array makes
        # skimage interpret the *height axis as a channel* and collapse the image
        # to a single interpolated row — which still looks like a plausible
        # `(1, H, W)` tensor in range, but destroys the content (found by checking
        # the checkpoint's own AUC on its training domain: 0.49 vs 0.80).
        normalised = xrv.datasets.normalize(image, max_value, reshape=True)
        return self._resize(normalised)
