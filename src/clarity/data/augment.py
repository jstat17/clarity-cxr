import numpy as np
from math import ceil
from scipy import ndimage
from collections.abc import Callable
from functools import partial

import cv2

from clarity.data.transform import (
    resize_3d,
    resize,
    rotate as rotate_about_center,
)

MAX_LAM = 10**5

# augmentations that only move geometry (shared by image and mask paths)
SIMILARITY_AUGMENTATIONS = {"random_rotate", "random_crop_resize"}


def rescale(
    val: np.ndarray | float,
    range_min: float,
    range_max: float,
    target_min: float,
    target_max: float,
) -> np.ndarray:
    """Rescale value/s from [range_min, range_max] to [target_min, target_max]

    Args:
        val (np.ndarray | float): Value/s to rescale
        range_min (float): Input range minimum
        range_max (float): Input range maximum
        target_min (float): Output range minimum
        target_max (float): Output range maximum

    Returns:
        np.ndarray: Rescaled value/s in [target_min, target_max]
    """
    return (val - range_min) / (range_max - range_min) * (target_max - target_min) + target_min


def normalize(val: np.ndarray) -> np.ndarray:
    val = rescale(val, range_min=val.min(), range_max=val.max(), target_min=0.0, target_max=1.0)
    return val


def dilate(arr: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Dilate a boolean numpy array (foreground is True).

    Parameters:
        arr (np.ndarray): A boolean numpy array.
        kernel_size (int or tuple of int): Size of the square (or rectangular) kernel.
            If an integer is provided, a square kernel of that size will be used.
        iterations (int): Number of dilation iterations.

    Returns:
        np.ndarray: The dilated boolean NumPy array.
    """
    # create structuring element (kernel) for morphological dilation
    structure = np.ones((kernel_size, kernel_size), dtype=bool)

    # perform the dilation
    arr = ndimage.binary_dilation(arr, structure=structure, iterations=iterations)

    return arr


def get_random_crop_bounds(
    image: np.ndarray, frac_height: float, frac_width: float
) -> tuple[int, int, int, int]:
    # get size of crop
    if len(image.shape) == 3:
        height, width = image.shape[1:]
    else:
        height, width = image.shape

    crop_pixels_height = ceil(height * frac_height)
    crop_pixels_width = ceil(width * frac_width)

    # get bounds of crop
    height_min = 0
    height_max = height - crop_pixels_height
    if height_min >= height_max:
        height_max = height_min + 1

    width_min = 0
    width_max = width - crop_pixels_width
    if width_min >= width_max:
        width_max = width_min + 1

    # sample random position of crop
    row_start = np.random.randint(height_min, height_max)
    row_end = row_start + crop_pixels_height

    col_start = np.random.randint(width_min, width_max)
    col_end = col_start + crop_pixels_width

    return row_start, col_start, row_end, col_end


def crop(
    image: np.ndarray, row_start: int, col_start: int, row_end: int, col_end: int
) -> np.ndarray:
    if len(image.shape) == 3:
        image_crop = image[:, row_start:row_end, col_start:col_end]
    else:
        image_crop = image[row_start:row_end, col_start:col_end]

    return image_crop


def random_crop_resize(image: np.ndarray, frac_min: float, frac_max: float) -> np.ndarray:
    frac = rescale(
        val=np.random.random(),
        range_min=0.0,
        range_max=1.0,
        target_min=frac_min,
        target_max=frac_max,
    )

    row_start, col_start, row_end, col_end = get_random_crop_bounds(
        image=image, frac_height=frac, frac_width=frac
    )

    # crop to selected region
    if len(image.shape) == 3:
        original_shape = image.shape[1:]
    else:
        original_shape = image.shape

    image = crop(
        image=image, row_start=row_start, col_start=col_start, row_end=row_end, col_end=col_end
    )

    # resize image back to original shape
    if len(image.shape) == 3:
        image = resize_3d(image, original_shape)
    else:
        image = resize(image, original_shape)

    return image


def random_poisson_noise(image: np.ndarray) -> np.ndarray:
    # get original pixel value range as target
    orig_min = image.min()
    orig_max = image.max()

    # rescale image to [0, 1]
    image = rescale(image, range_min=orig_min, range_max=orig_max, target_min=0.0, target_max=1.0)

    # generate and add noise
    image = _sample_poisson_noise(image)

    # rescale to original pixel value range
    range_min = image.min()
    range_max = image.max()

    image = rescale(
        val=image,
        range_min=range_min,
        range_max=range_max,
        target_min=orig_min,
        target_max=orig_max,
    )
    image = np.where(np.isnan(image), 0, image)
    image = np.clip(image, orig_min, orig_max)

    return image


def _sample_poisson_noise(image: np.ndarray) -> np.ndarray:
    # sample gain (base-10 logarithmic scale)
    gain_min_log = 0.0
    gain_max_log = 3.0
    gain_log = rescale(
        val=np.random.random(),
        range_min=0.0,
        range_max=1.0,
        target_min=gain_min_log,
        target_max=gain_max_log,
    )
    gain = 10**gain_log

    # sample noise based on image pixel values
    lam = gain * np.abs(image)
    lam = np.where(np.isnan(lam), 0, lam)
    lam = np.clip(lam, 0, MAX_LAM)

    noise = np.random.poisson(lam) / gain
    noise = np.where(np.isnan(noise), 0, noise)
    noise = np.clip(noise, 0, 1)

    return noise


def random_poisson_noise_cropped_region(
    image: np.ndarray, frac_min: float, frac_max: float
) -> np.ndarray:
    frac_height = rescale(
        val=np.random.random(),
        range_min=0.0,
        range_max=1.0,
        target_min=frac_min,
        target_max=frac_max,
    )
    frac_width = rescale(
        val=np.random.random(),
        range_min=0.0,
        range_max=1.0,
        target_min=frac_min,
        target_max=frac_max,
    )

    row_start, col_start, row_end, col_end = get_random_crop_bounds(
        image=image, frac_height=frac_height, frac_width=frac_width
    )

    # get cropped region and add poisson noise
    image_crop = crop(
        image=image.copy(),
        row_start=row_start,
        col_start=col_start,
        row_end=row_end,
        col_end=col_end,
    )
    image_crop = random_poisson_noise(image_crop)

    # overwrite cropped region with poisson noise version
    if len(image.shape) == 3:
        image[:, row_start:row_end, col_start:col_end] = image_crop
    else:
        image[row_start:row_end, col_start:col_end] = image_crop

    return image


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image about its center (OpenCV cubic interpolation, zero
    border). Accepts (H, W) or channel-first (C, H, W) input; a multi-channel
    image is warped in a single batched call."""
    return rotate_about_center(image, angle)


def random_rotate(image: np.ndarray, angle_min: float, angle_max: float) -> np.ndarray:
    angle = rescale(
        val=np.random.random(),
        range_min=0.0,
        range_max=1.0,
        target_min=angle_min,
        target_max=angle_max,
    )

    image_rotate = rotate(image, angle)
    return image_rotate


def change_brightness(image: np.ndarray, brightness: float) -> np.ndarray:
    orig_min = image.min()
    orig_max = image.max()
    orig_range = orig_max - orig_min

    br = brightness * orig_range
    image_bright = np.clip(image + br, a_min=orig_min, a_max=orig_max)

    return image_bright


def random_brightness(
    image: np.ndarray, brightness_min: float, brightness_max: float
) -> np.ndarray:
    brightness = rescale(
        val=np.random.random(),
        range_min=0.0,
        range_max=1.0,
        target_min=brightness_min,
        target_max=brightness_max,
    )

    image_bright = change_brightness(image, brightness)
    return image_bright


def generate_augmentations_list(
    rotate_prob: float,
    crop_prob: float,
    brightness_prob: float,
    noise_prob: float,
    n_noise: int,
    rotate_angle_min: float = -15.0,
    rotate_angle_max: float = 15.0,
    crop_frac_min: float = 0.7,
    crop_frac_max: float = 1.0,
    brightness_min: float = -0.5,
    brightness_max: float = 0.5,
    poisson_frac_min: float = 0.0,
    poisson_frac_max: float = 1.0,
) -> list[Callable]:
    augmentations = []

    if np.random.random() < rotate_prob:
        func_aug = partial(random_rotate, angle_min=rotate_angle_min, angle_max=rotate_angle_max)
        augmentations.append(func_aug)

    if np.random.random() < crop_prob:
        func_aug = partial(random_crop_resize, frac_min=crop_frac_min, frac_max=crop_frac_max)
        augmentations.append(func_aug)

    if np.random.random() < brightness_prob:
        func_aug = partial(
            random_brightness, brightness_min=brightness_min, brightness_max=brightness_max
        )
        augmentations.append(func_aug)

    if np.random.random() < noise_prob:
        func_aug = partial(
            random_poisson_noise_cropped_region,
            frac_min=poisson_frac_min,
            frac_max=poisson_frac_max,
        )
        n_noise_aug = np.random.randint(1, n_noise + 1)
        for _ in range(n_noise_aug):
            augmentations.append(func_aug)

    return augmentations


def apply_augmentations(image: np.ndarray, augmentations: list[Callable]) -> np.ndarray:
    for augmentation in augmentations:
        image = augmentation(image)

    return image


def apply_similarity_augmentations(image: np.ndarray, augmentations: list[Callable]) -> np.ndarray:
    for augmentation in augmentations:
        augmentation: partial
        if augmentation.func in [random_rotate, random_crop_resize]:
            image = augmentation(image)

    return image


def sample_similarity_transforms(augmentations: list[Callable]) -> list[tuple[str, float]]:
    """Replay the rotation/crop draws that apply_similarity_augmentations would
    consume for one mask, as ('rotate', angle) / ('crop', frac) pairs.

    The per-mask augmentation loop resets the RNG state before every mask, so
    every mask receives exactly this draw sequence; sampling it once and
    applying it to all masks together is equivalent.
    """
    transforms = []
    for augmentation in augmentations:
        augmentation: partial
        if augmentation.func is random_rotate:
            angle = rescale(
                val=np.random.random(),
                range_min=0.0,
                range_max=1.0,
                target_min=augmentation.keywords["angle_min"],
                target_max=augmentation.keywords["angle_max"],
            )
            transforms.append(("rotate", float(angle)))
        elif augmentation.func is random_crop_resize:
            frac = rescale(
                val=np.random.random(),
                range_min=0.0,
                range_max=1.0,
                target_min=augmentation.keywords["frac_min"],
                target_max=augmentation.keywords["frac_max"],
            )
            transforms.append(("crop", float(frac)))

    return transforms


def crop_bounds(shape: tuple[int, int], frac: float) -> tuple[int, int]:
    """Sample the crop size for a square image at the given fraction, using
    the same ceil-based sizing and minimum-size guard as get_random_crop_bounds.

    Returns:
        tuple[int, int]: (crop_height, crop_width)
    """
    height, width = shape
    crop_h = min(ceil(height * frac), height)
    crop_w = min(ceil(width * frac), width)
    return crop_h, crop_w


def batched_crop_resize(stack: np.ndarray, frac: float) -> np.ndarray:
    """Crop each channel-last plane of an (H, W, K) uint8 stack to the given
    fraction (random shared offset, cv2 bilinear resize back to full size).

    Applies the same geometry to all planes at once; per-plane calls would
    produce identical results but pay OpenCV call overhead K times.

    Args:
        stack (np.ndarray): Channel-last stack of shape (H, W, K)
        frac (float): Crop fraction in (0, 1]

    Returns:
        np.ndarray: Crop-resized stack of shape (H, W, K)
    """
    height, width = stack.shape[:2]
    crop_h, crop_w = crop_bounds((height, width), frac)

    h_max = max(height - crop_h, 1)
    w_max = max(width - crop_w, 1)
    row_start = np.random.randint(0, h_max)
    col_start = np.random.randint(0, w_max)

    crop = stack[row_start : row_start + crop_h, col_start : col_start + crop_w, :]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    # cv2.resize drops a trailing singleton channel dimension
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return resized


def fuse_multiply(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    arr_fused = arr1 * arr2
    arr_fused = rescale(
        val=arr_fused,
        range_min=arr_fused.min(),
        range_max=arr_fused.max(),
        target_min=0.0,
        target_max=1.0,
    )

    return arr_fused
