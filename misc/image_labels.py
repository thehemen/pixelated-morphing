import os
import cv2
import glob
import numpy as np
from misc.misc import make_dir
from misc.misc import finetune_mask

image_dir_name = 'labeled_assets/'

pair_classes = ['Eye', 'Ear', 'Eyebrow']
all_classes = ['Hair', 'Helmet', 'Eye', 'Nose', 'Mouth', 'Ear', 'Eyebrow', 'Head']

def is_pair_class(class_name):
    """Check if class is paired (eye, ear, eyebrow)."""
    return class_name in pair_classes

def assign_image_areas(labeled_im, region_imgs):
    """Assign images to the labeled image regions.

    Keyword arguments:
    labeled_im -- labeled image
    region_imgs -- region images
    """
    for class_name in all_classes:
        if class_name not in labeled_im.regions.keys():
            continue

        if not labeled_im.regions[class_name].is_paired:
            img_area = region_imgs[class_name]
            labeled_im.regions[class_name].image_area = img_area
        else:
            regions = labeled_im.regions[class_name].get_regions()

            for i, region in enumerate(regions):
                img_name = f'{class_name}_{i + 1}'
                img_area = region_imgs[img_name]
                region.image_area = img_area

def load_images(imgs):
    """Load PNG images as labeled image regions."""
    for labeled_im in imgs.values():
        region_imgs = {}
        image_name = labeled_im.image_name
        img_area_names = sorted(glob.glob(f'{image_dir_name}{image_name}/*.png'))

        for img_area_name in img_area_names:
            img_area_name_no_dir = img_area_name.split('/')[-1].split('.')[0]
            img_area = cv2.imread(img_area_name, cv2.IMREAD_UNCHANGED)
            region_imgs[img_area_name_no_dir] = img_area

        assign_image_areas(labeled_im, region_imgs)

def fill_image_area(im_region, image):
    """Fill image areas with its regions.

    Keyword arguments:
    im_region -- image_region
    image -- cv2 image
    """
    bbox = im_region.bbox

    if not im_region.points:
        y1, x1 = bbox.y1, bbox.x1
        y2, x2 = bbox.y2, bbox.x2

        im_region.image_area = image[y1: y2, x1: x2].copy()
        image[y1: y2, x1: x2] = 0
    else:
        # Add an image by the polygon mask.
        points = np.array([im_region.points])

        height = image.shape[0]
        width = image.shape[1]

        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.fillPoly(mask, points, (255))
        mask = finetune_mask(mask)
        res = cv2.bitwise_and(image, image, mask=mask)

        x1, y1 = bbox.x1, bbox.y1
        w, h = bbox.width, bbox.height

        im_region.image_area = res[y1: y1 + h, x1: x1 + w]
        image[mask > 0] = 0

def fill_image_areas(labeled_im):
    """Fill image areas for all classes."""
    for class_name in all_classes:
        if class_name not in labeled_im.regions.keys():
            continue

        if not labeled_im.regions[class_name].is_paired:
            fill_image_area(labeled_im.regions[class_name], labeled_im.image)
        else:
            regions = labeled_im.regions[class_name].get_regions()

            for region in regions:
                fill_image_area(region, labeled_im.image)

def save_images(imgs):
    """Save labeled image regions as PNG images."""
    make_dir(image_dir_name)

    for labeled_im in imgs.values():
        image_name = labeled_im.image_name
        sub_dir_name = f'{image_dir_name}{image_name}/'
        os.mkdir(sub_dir_name)

        for class_name, region in labeled_im.regions.items():
            if not region.is_paired:
                cv2.imwrite(f'{sub_dir_name}{class_name}.png', region.image_area)
            else:
                regions = region.get_regions()

                for i, region in enumerate(regions):
                    cv2.imwrite(f'{sub_dir_name}{class_name}_{i + 1}.png', region.image_area)
