import cv2
import numpy as np
from copy import deepcopy
from skimage.measure import block_reduce
from misc.misc import unpixelate, apply_over_channels
from misc.colors import get_color_transfer

classes = ['Head', 'Eye', 'Nose', 'Mouth', 'Hair', 'Helmet', 'Ear', 'Eyebrow']

def get_aligned_image(img, align_width):
    """Get an image aligned by a meta-pixel's width."""
    img = block_reduce(img, block_size=(align_width, align_width, 1), func=np.mean)
    img = apply_over_channels(unpixelate, img, align_width).astype(np.uint8)
    return img

class StyleTransfer:
    """
        StyleTransfer implements a style transfer between images.
    """
    def __init__(self, img1, img2, align_width):
        """StyleTransfer constructor.

        Fields:
        img1 -- an image for which the style is transferred
        img2 -- an image from which the style is transferred
        align_width -- a meta-pixel's width that is used to align an image
        """
        self.img1 = img1
        self.img2 = img2

        self.align_width = align_width

    def apply(self):
        """Apply a style transfer between images.

        A color transfer technique is used
        when the first image's class is not given
        in the second image.

        The first image is defined as the destination
        and the second image's "Head" region is defined
        as the source of the color transfer.

        The reason is that regions of an image may have
        different color styles so that it makes challenging
        to retrieve a style from all of them.

        That's why the only "Head" image region is used
        to apply a color transfer.
        """
        img_styled = deepcopy(self.img1)
        source = self.img2['Head'].image_area

        for class_name in classes:
            if class_name not in self.img1.get_classes():
                continue

            imgs_1 = self.img1[class_name].get_regions()
            bboxes = [img.bbox for img in imgs_1]

            is_pair_class = len(imgs_1) == 2

            if class_name not in self.img2.get_classes():
                imgs = [img.image_area for img in imgs_1]
                is_color_transfer = True
            else:
                imgs_2 = self.img2[class_name].get_regions()
                imgs = [img.image_area for img in imgs_2]
                is_color_transfer = False

            for i, (img_area, bbox) in enumerate(zip(imgs, bboxes)):
                width, height = bbox.width, bbox.height

                if is_color_transfer:
                    # Apply a color transfer technique.
                    img_new = img_area.copy()
                    img_new[:, :, :3] = get_color_transfer(source, img_new)
                else:
                    # Apply a resized version of the image area.
                    img_new = cv2.resize(img_area, (width, height))

                img_new = get_aligned_image(img_new, self.align_width)

                if not is_pair_class:
                    img_styled[class_name].image_area = img_new
                    img_styled[class_name].bbox = bbox
                else:
                    img_styled[class_name][i].image_area = img_new
                    img_styled[class_name][i].bbox = bbox

        # Return an image with an updated style.
        return img_styled
