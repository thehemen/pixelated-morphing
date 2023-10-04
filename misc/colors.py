import cv2
import math
import warnings
import numpy as np
from skimage.color import rgb2lab, lab2lch, lch2lab, lab2rgb

def get_color_transfer(source, destination):
    """Transfer colors from source to destination image.

    Keyword arguments:
    source -- source image from which the colors are transferred
    destination - destination image for which the colors are transferred
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2Lab).astype(float)
    destination = cv2.cvtColor(destination, cv2.COLOR_BGR2Lab).astype(float)

    src_mean_Lab = source.mean(axis=(0, 1))
    dest_mean_Lab = destination.mean(axis=(0, 1))

    src_std_Lab = source.std(axis=(0, 1))
    dest_std_Lab = destination.std(axis=(0, 1)) + 1e-9

    result_Lab = destination - dest_mean_Lab
    result_Lab = (src_std_Lab / dest_std_Lab) * result_Lab

    result_Lab += src_mean_Lab
    result_Lab = np.clip(result_Lab, 0, 255)

    result = cv2.cvtColor(result_Lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return result

def get_adjusted_channel(channel, value, color_range):
    """Adjust color value in range [min_color; max_color]."""
    min_color, max_color = color_range

    if value < 0.0:
        channel += (channel - min_color) * value
    else:
        channel += (max_color - channel) * value

    return channel

def adjust_colors(img, luminance, chroma, hue):
    """Adjust an image's colors by LCh values.

    Keyword arguments:
    luminance -- the luminance value
    chroma -- the chroma value
    hue -- the hue value
    """
    img_new = img.copy()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    lab_img = rgb2lab(rgb_img)
    lch_img = lab2lch(lab_img)

    lch_img[:, :, 0] = get_adjusted_channel(lch_img[:, :, 0], luminance, color_range=[0.0, 100.0])
    lch_img[:, :, 1] = get_adjusted_channel(lch_img[:, :, 1], chroma, color_range=[0.0, 100.0])
    lch_img[:, :, 2] = get_adjusted_channel(lch_img[:, :, 2], hue, color_range=[0.0, 2 * math.pi])

    lab_img_new = lch2lab(lch_img)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rgb_img_new = lab2rgb(lab_img_new)

    rgb_img_new = (rgb_img_new * 255.0).round().astype(np.uint8)
    img_new[:, :, :3] = cv2.cvtColor(rgb_img_new, cv2.COLOR_RGB2BGR)
    return img_new
