import torch as th
import numpy as np
import cv2
from math import ceil, floor
from collections.abc import Callable
from typing import Any, Type


class InvalidInterpolationMethod(Exception):
    """Raised when an invalid interpolation method is used."""

    pass


def resize(image: np.ndarray, shape: tuple[int, int], method: str = "bilinear") -> np.ndarray:
    """Resize an image to a new size.

    Args:
        image (np.ndarray): Image of shape (height, width)
        shape (tuple[int, int]): New image size as (new_height, new_width)
        method (str): Sampling method. Defaults to 'bilinear'

    Returns:
        np.ndarray: Resized image of shape (new_height, new_width)

    Raises:
        InvalidInterpolationMethod: If a method other than nearest, bilinear, area,
            bicubic or lanczos is given
    """
    match method:
        case "nearest":
            interpolation = cv2.INTER_NEAREST
        case "bilinear":
            interpolation = cv2.INTER_LINEAR
        case "area":
            interpolation = cv2.INTER_AREA
        case "bicubic":
            interpolation = cv2.INTER_CUBIC
        case "lanczos":
            interpolation = cv2.INTER_LANCZOS4
        case _:
            raise InvalidInterpolationMethod(f"Interpolation method incorrect: {method}")

    return cv2.resize(image, shape[::-1], interpolation=interpolation)


def resize_3d(image: np.ndarray, shape: tuple[int, int], method: str = "bilinear") -> np.ndarray:
    """Resize a 3D image (with channels dimension) to a new size.

    All channels are resized in a single batched OpenCV call, which is
    byte-identical to resizing each channel separately but avoids the
    per-channel Python loop.

    Args:
        image (np.ndarray): Image of shape (channels, height, width)
        shape (tuple[int, int]): New 2D image size dimensions as (new_height, new_width)
        method (str): Sampling method. Defaults to 'bilinear'

    Returns:
        np.ndarray: Resized image of shape (channels, new_height, new_width)
    """
    moved = np.ascontiguousarray(np.moveaxis(image, 0, -1))
    resized = resize(moved, shape, method=method)
    return np.ascontiguousarray(np.moveaxis(resized, -1, 0))


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image about its center with OpenCV cubic interpolation and a
    zero border, matching scipy.ndimage.rotate's direction and center.

    A channels-first (C, H, W) image is warped in one batched call, which is
    identical to warping each channel separately but avoids the per-channel
    Python loop.

    Args:
        image (np.ndarray): Image of shape (H, W) or (C, H, W)
        angle (float): Rotation angle in degrees (counter-clockwise)

    Returns:
        np.ndarray: Rotated image, same shape and dtype as the input
    """
    if image.ndim == 3:
        moved = np.ascontiguousarray(np.moveaxis(image, 0, -1))
        height, width = moved.shape[:2]
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(
            moved,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.ascontiguousarray(np.moveaxis(rotated, -1, 0))

    height, width = image.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_affine_channels(stack: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """warpAffine an (H, W, K) channel-last stack in chunks of at most 4
    channels (OpenCV caps warpAffine at 4 channels with cubic interpolation).

    Args:
        stack (np.ndarray): Channel-last stack of shape (H, W, K)
        matrix (np.ndarray): 2x3 affine transformation matrix

    Returns:
        np.ndarray: Warped stack of shape (H, W, K)
    """
    height, width, channels = stack.shape
    if channels == 1:
        out = cv2.warpAffine(
            stack[:, :, 0],
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return out[:, :, None]

    outs = []
    for start in range(0, channels, 4):
        outs.append(
            cv2.warpAffine(
                stack[:, :, start : start + 4],
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        )
    return np.concatenate(outs, axis=-1)


def crop_square_equal(image: np.ndarray) -> np.ndarray:
    """Crops an input image to a square shape by shortening the longest side equally.

    Args:
        image (np.ndarray): Input image as a NumPy array with shape (H, W) or (H, W, C),
            where H is height, W is width, and C is the number of channels (3 for RGB, 4 for RGBA).

    Returns:
        np.ndarray: Cropped square image with shape (S, S) or (S, S, C), where S is the
            smaller of the original height and width.

    Raises:
        ValueError: If the image does not have at least 2 dimensions or if either dimension
            is less than 1.
    """
    # check that the image has at least 2 dimensions
    if image.ndim < 2:
        raise ValueError("Input image must have at least 2 dimensions (height and width).")

    # extract height (H) and width (W) from the image shape
    H, W = image.shape[:2]

    # check that dimensions are positive
    if H < 1 or W < 1:
        raise ValueError("Image dimensions must be positive integers.")

    # if already square, return the original image
    if H == W:
        return image

    # determine the target square size (the smaller dimension)
    target_size = min(H, W)

    # crop the longer dimension symmetrically
    if H > W:
        # crop rows (height) to match width
        start_row = (H - target_size) // 2
        end_row = start_row + target_size
        cropped_image = image[start_row:end_row, ...]

    else:  # W > H
        # crop columns (width) to match height
        start_col = (W - target_size) // 2
        end_col = start_col + target_size
        cropped_image = image[:, start_col:end_col, ...]

    return cropped_image


def border_pad_image(image: np.ndarray, border_padding: int) -> np.ndarray:
    if border_padding == 0:
        return image

    # create blank image
    image_pad = np.zeros_like(image)

    # handle 2D and 3D images
    if len(image.shape) == 3:
        C, H, W = image.shape
    else:
        C, H, W = (None, *image.shape)

    # get downsampled image shape
    new_image_shape = (int(H - 2 * border_padding), int(W - 2 * border_padding))

    # downsample image and fit within bordered region
    if C is not None:
        image = resize_3d(image, new_image_shape)
        image_pad[
            :,
            border_padding : new_image_shape[0] + border_padding,
            border_padding : new_image_shape[1] + border_padding,
        ] = image

    else:
        image = resize(image, new_image_shape)
        image_pad[
            border_padding : new_image_shape[0] + border_padding,
            border_padding : new_image_shape[1] + border_padding,
        ] = image

    return image_pad


def manipulate_boolean_mask(
    mask: np.ndarray, func: Callable, thr: int = 127, **kwargs
) -> np.ndarray:
    # convert bool to uint8
    mask = mask.astype(np.uint8) * 255
    # apply function
    mask = func(mask, **kwargs)
    # convert back to bool
    mask = mask > thr

    return mask


def get_bounding_box_from_mask(mask_batch: th.Tensor) -> th.Tensor:
    """Fit a bounding box around each mask in batch, and return the co-ordinates.

    Args:
        mask_batch (th.Tensor): Binary mask of shape (batch_size, height, width)

    Returns:
        th.Tensor: Bounding box coordinates (x1, y1, x2, y2) for each mask in the batch\
    Where x1, x2 are the row indices, and y1, y2 are the column indices
    """
    # get the indices of non-zero elements in the mask
    non_zero_indices = th.nonzero(mask_batch, as_tuple=True)

    # initialize the bounding box coordinates
    batch_size = mask_batch.shape[0]
    bounding_boxes = th.zeros((batch_size, 4), dtype=th.int32)

    # iterate over each mask in the batch
    for i in range(batch_size):
        # get the indices of non-zero elements in the current mask
        indices = (
            non_zero_indices[1][non_zero_indices[0] == i],
            non_zero_indices[2][non_zero_indices[0] == i],
        )

        # calculate the bounding box coordinates
        if len(indices[0]) > 0:
            x1, y1 = th.min(indices[0]), th.min(indices[1])
            x2, y2 = th.max(indices[0]), th.max(indices[1])
            bounding_boxes[i] = th.tensor([x1, y1, x2, y2])
        else:
            # if the mask is empty, set the bounding box coordinates to 0
            bounding_boxes[i] = th.tensor([0, 0, 0, 0])

    return bounding_boxes


def scale_bounding_box(
    x1: int, y1: int, x2: int, y2: int, shape_old: tuple[int, int], shape_new: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Scale a set of bounding box co-ordinates from an old to a new image shape

    Args:
        x1 (int): Start row index of the bounding box
        y1 (int): Start column index of the bounding box
        x2 (int): End row index of the bounding box
        y2 (int): End column index of the bounding box
        shape_old (tuple[int, int]): Shape of the original image
        shape_new (tuple[int, int]): Shape of the new image

    Returns:
        tuple[int, int, int, int]: Scaled bounding box co-ordinates
    """
    height_old, width_old = shape_old
    height_new, width_new = shape_new
    scale_y = width_new / width_old
    scale_x = height_new / height_old

    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    # Ensure the bounding box coordinates are within the new image bounds
    x1 = max(0, min(x1, height_new - 1))
    y1 = max(0, min(y1, width_new - 1))
    x2 = max(0, min(x2, height_new - 1))
    y2 = max(0, min(y2, width_new - 1))

    return x1, y1, x2, y2


def make_bounding_box_square(
    x1: int, y1: int, x2: int, y2: int, shape: tuple[int, int], allowance: float = 0.005
) -> tuple[int, int, int, int]:
    """Make an existing bounding box square, and ensure it fits on its reference image.

    Args:
        x1 (int): Start row index of the bounding box
        y1 (int): Start column index of the bounding box
        x2 (int): End row index of the bounding box
        y2 (int): End column index of the bounding box
        shape (tuple[int, int]): Shape of the reference image
        allowance (float, optional): Growth of the major dimension in case of segmentation errors. Defaults to 0.005.

    Returns:
        tuple[int, int, int, int]: Squared bounding box co-ordinates
    """
    # subtract 1 to be in index scale
    img_height, img_width = shape
    img_height -= 1
    img_width -= 1

    height = x2 - x1
    width = y2 - y1

    # do nothing if already square
    if height == width:
        return x1, y1, x2, y2

    # set alias variables for the two cases
    elif height > width:
        w = img_height
        z = img_width
        w1, w2 = x1, x2
        z1, z2 = y1, y2
        ordering = True

    elif height < width:
        w = img_width
        z = img_height
        w1, w2 = y1, y2
        z1, z2 = x1, x2
        ordering = False

    # grow w to the allowance
    growth = allowance * (w2 - w1) / 2
    w1 = floor(w1 - growth)
    w2 = ceil(w2 + growth)

    # match z with w
    growth = ((w2 - w1) - (z2 - z1)) / 2
    z1 = floor(z1 - growth)
    z2 = ceil(z2 + growth)

    # shift bounding box until it fits in the image
    if z1 < 0 and z2 <= z:
        z2 = z2 - z1
        z1 = 0
    elif z1 >= 0 and z2 > z:
        z1 = z1 - (z2 - z)
        z2 = z

    if w1 < 0 and w2 <= w:
        w2 = w2 - w1
        w1 = 0
    elif w1 >= 0 and w2 > w:
        w1 = w1 - (w2 - w)
        w2 = w

    # if the bounding box dimensions are larger than the image then downscale the whole box
    if (w2 - w1) > w:
        diminish = ((w2 - w1) - w) / 2
        w1 = 0
        w2 = w
        z1 = ceil(z1 + diminish)
        z2 = floor(z2 - diminish)

    if (z2 - z1) > z:
        diminish = ((z2 - z1) - z) / 2
        z1 = 0
        z2 = z
        w1 = ceil(w1 + diminish)
        w2 = floor(w2 - diminish)

    # if from rounding there are some extra pixels
    extra_pixels = abs((w2 - w1) - (z2 - z1))
    # first make it so that there are an even number of extra pixels
    if extra_pixels % 2 == 1:
        if (w2 - w1) > (z2 - z1):
            w1 += 1
        else:
            z1 += 1

    # then remove the extra pixels on either side of the bounding box
    if extra_pixels % 2 == 0:
        extra_pixels_div_2 = int(extra_pixels / 2)
        if (w2 - w1) > (z2 - z1):
            w1 += extra_pixels_div_2
            w2 -= extra_pixels_div_2
        else:
            z1 += extra_pixels_div_2
            z2 -= extra_pixels_div_2

    if ordering:
        return w1, z1, w2, z2
    else:
        return z1, w1, z2, w2


def fnv1a_32(val: Any) -> int:
    """FNV-1a 32-bit hashing function."""
    val = str(val)

    # 32-bit FNV offset basis and prime
    FNV_prime = 0x01000193
    FNV_offset_basis = 0x811C9DC5

    # Convert the string into bytes (UTF-8 encoding)
    hash_value = FNV_offset_basis
    for byte in val.encode("utf-8"):
        hash_value ^= byte
        hash_value *= FNV_prime
        hash_value &= 0xFFFFFFFF  # Ensure it's a 32-bit number

    return hash_value


def get_max_value(arr: np.ndarray) -> int:
    return np.iinfo(arr.dtype).max


def encode_multiclass_one_hot(
    possible_labels: list[str], labels: list[str], dtype: Type = np.float32
) -> np.ndarray:
    # create a set for faster lookup
    possible_labels_set = set(labels)

    # initialize the multi-hot encoded vector
    vec = np.zeros(len(possible_labels), dtype=dtype)

    # iterate through possible labels and set the corresponding index in the vector
    for i, label in enumerate(possible_labels):
        if label in possible_labels_set:
            vec[i] = 1

    return vec
