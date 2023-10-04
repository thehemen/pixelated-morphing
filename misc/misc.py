import os
import cv2
import shutil
import numpy as np
from functools import reduce
from collections import Counter

def finetune_mask(mask, cell_size=4, threshold=0.5):
    """Finetune an image mask by applying 0 or max value, that depends on threshold.

    Keyword arguments:
    cell_size -- the size of the cell
    threshold -- the value of threshold
    """
    height, width = mask.shape
    max_pixel_value = np.max(mask)
    pixel_threshold = (cell_size * cell_size) * threshold

    row_num = int(height // cell_size)
    col_num = int(width // cell_size)

    for i in range(row_num):
        for j in range(col_num):
            y1, y2 = i * cell_size, (i + 1) * cell_size
            x1, x2 = j * cell_size, (j + 1) * cell_size

            pixel_value = np.sum(mask[y1: y2, x1: x2])
            pixel_value = int(pixel_value // max_pixel_value)

            if pixel_value > pixel_threshold:
                pixel_value = max_pixel_value
            else:
                pixel_value = 0

            mask[y1: y2, x1: x2] = pixel_value

    return mask

def x_round(x, cell_size=4):
    """Round a coordinate value by the cell size."""
    return int(round(x / cell_size) * cell_size)

def get_equal_values(values_1, values_2):
    """Get coordinate names which values are equal."""
    values = []

    for key_1, value_1 in values_1.items():
        axis_1, number_1 = key_1[0], int(key_1[1])

        for key_2, value_2 in values_2.items():
            axis_2, number_2 = key_2[0], int(key_2[1])

            if (axis_1 == axis_2) and (value_1 == value_2):
                values.append(tuple((key_1, key_2)))

    return values

def gcd(a, b):
    """Get the Greatest Common Divisor (GCD) of two values."""
    if b == 0:
        return a
    else:
        return gcd(b, a % b)

def get_align_width(img, n=8):
    """Get the align width of an image."""
    img = img.astype(int)
    img_diff = img[:-1, :-1] - img[1:, 1:]

    indices = np.nonzero(img_diff != 0)[:2]
    indices = np.concatenate(indices) + 1

    values = Counter(indices).most_common(n=n)
    values = [x[0] for x in values]

    width = reduce(gcd, values)
    return width

def pixelate(a, shape):
    """Pixelate (compress) an image by shape."""
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1).astype(np.uint8)

def unpixelate(a, width):
    """Unpixelate (uncompress) an image by width."""
    return a.repeat(width, 0).repeat(width, 1)

def apply_over_channels(func, img, shape):
    """Apply the function over each channel of the image's shape."""
    return cv2.merge([func(x, shape) for x in cv2.split(img)])

def add_alpha_images(img_1, img_2):
    """Add two semi-transparent images by their alpha channels."""
    alpha_img_1 = img_1[:, :, 3] / 255.0
    alpha_img_2 = img_2[:, :, 3] / 255.0

    for color in range(3):
        img_1[:, :, color] = alpha_img_2 * img_2[:, :, color] + alpha_img_1 * img_1[:, :, color] * (1 - alpha_img_2)

    img_1[:, :, 3] = (1 - (1 - alpha_img_2) * (1 - alpha_img_1)) * 255
    return img_1

def get_random_beta(low=0.0, high=1.0, alpha=3.0, beta=3.0):
    """Get random value from beta distribution."""
    beta_value = np.random.beta(alpha, beta)
    beta_value = (high - low) * beta_value - low
    return beta_value

def make_dir(dir_name):
    """Make a folder or overwrite if it already exists."""
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    os.mkdir(dir_name)
