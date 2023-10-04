import cv2
import numpy as np

def get_color_diff(im_pixels):
    """Get a color difference between the top-left color and other colors."""
    im_color = im_pixels[0, 0].astype(int)
    color_diff = im_pixels - im_color
    return color_diff

def get_mean_colors(im_region):
    """Get mean colors of the 3x3 image region."""
    color_diffs = []

    for x in range(2):
        for y in range(2):
            im_pixels = im_region[x: x + 2, y: y + 2]
            color_diff = get_color_diff(im_pixels)
            color_diffs.append(color_diff)

    mean_colors = np.mean(color_diffs, axis=0)
    return mean_colors

def get_mixed_colors(colors):
    """Get mixed colors by subtraction and averaging colors."""
    all_colors = []

    for x in range(2):
        for y in range(2):
            all_colors.append(colors - colors[x, y])

    mixed_colors = np.mean(all_colors, axis=0).round().astype(int)
    return mixed_colors

def clip_colors(colors, im_color):
    """Clip colors to fit their values in the range [0; 255]."""
    im_color = im_color.astype(int)

    min_color = -im_color
    max_color = 255 - im_color

    colors = np.clip(colors, min_color, max_color)
    return colors

def apply_smooth_upscaling(im_region):
    """Apply smooth upscaling of an image region.

    Keyword arguments:
    im_region -- an image region
    """
    im_color = im_region[1, 1]
    colors = np.array([[im_color, im_color], [im_color, im_color]])

    mean_colors = get_mean_colors(im_region)
    mixed_colors = get_mixed_colors(mean_colors)
    updated_colors = clip_colors(mixed_colors, im_color)

    colors = (colors + updated_colors).astype(np.uint8)
    colors = cv2.medianBlur(colors, 3)
    return colors

def apply_random_upscaling(im_region, randomness=4.0):
    """Apply random upscaling of an image region.
    Color values are given from the dirichlet distribution.

    Keyword arguments:
    im_region -- an image region
    randomness -- a measure of randomness
    """
    colors = np.zeros((2, 2, 3), dtype=np.uint8)
    indices = np.ndindex(colors.shape[:2])

    for x, y in indices:
        im_colors = np.reshape(im_region[x: x + 2, y: y + 2], (4, 3))
        weights = np.random.dirichlet(np.ones(4) / randomness, size=1)[0]
        colors[x, y] = np.average(im_colors, weights=weights, axis=0)

    colors = cv2.medianBlur(colors, 3)
    return colors

def apply_upscaling(img_initial, img_upscaled, threshold=65.0):
    """Apply upscaling of an image.
    Smooth or random upscaling can be chosen depending on the threshold value.

    Keyword arguments:
    img_initial -- an initial image
    img_upscaled -- an upscaled image
    threshold -- a threshold value of how smooth is a color shift of an image
    """
    gray = cv2.cvtColor(img_initial, cv2.COLOR_BGRA2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    mean, std_dev = cv2.meanStdDev(laplacian)
    score = std_dev[0][0]

    indices = np.ndindex(img_initial.shape[:2])
    img_padded = cv2.copyMakeBorder(img_initial, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    for x, y in indices:
        x_i, y_i = x * 2, y * 2
        im_region = img_padded[x: x + 3, y: y + 3, :3]

        if score < threshold:
            colors = apply_smooth_upscaling(im_region)
        else:
            colors = apply_random_upscaling(im_region)

        img_upscaled[x_i: x_i + 2, y_i: y_i + 2, :3] = colors

    return img_upscaled
