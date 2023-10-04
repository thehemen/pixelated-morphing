import cv2
import json
from misc.misc import x_round, get_align_width
from misc.dependencies import find_dependencies
from misc.image_labels import load_images, fill_image_areas, is_pair_class
from pixel_art.bounding_box import BoundingBox
from pixel_art.labeled_im import ImRegion, LabeledIm

def read_images_and_labels(filename, already_saved=False):
    """Read images and labels that represent the location of image regions.

    Keyword arguments:
    filename -- a JSON filename
    already_saved -- a flag that shows were the images already saved or not
    """
    imgs = {}
    img_names = {}
    categories = {}

    # Load a JSON file with labels.
    with open(filename, 'r') as f:
        json_dict = json.load(f)

    # Assign each category a unique ID.
    for category in json_dict['categories']:
        categories[category['id']] = category['name']

    # Load images and save them as a dictionary with a key from the image name.
    for image_dict in json_dict['images']:
        image_id = image_dict['id']
        file_name = image_dict['file_name']
        image = cv2.imread(f'aligned_assets/{file_name}', cv2.IMREAD_UNCHANGED)
        width = get_align_width(image)
        image_name = file_name.split('.')[0]
        img_names[image_id] = image_name
        imgs[image_name] = LabeledIm(image_name, image, width)

    # Assign annotations to the images as the image regions.
    for annotation in json_dict['annotations']:
        image_id = annotation['image_id']
        class_name = categories[annotation['category_id']]

        x1, y1, width, height = annotation['bbox']
        x1, y1 = x_round(x1), x_round(y1)
        width, height = x_round(width), x_round(height)

        bounding_box = BoundingBox(x1, y1, width, height)
        points = None

        if annotation['segmentation']:
            points = annotation['segmentation'][0]
            points = [[x_round(x), x_round(y)] for x, y in zip(points[::2], points[1::2])]

        im_region = ImRegion(class_name, bounding_box, points)
        is_pair_region = is_pair_class(class_name)
        imgs[img_names[image_id]].add_region(im_region, is_pair_region)

    # Find dependencies between the image regions.
    for labeled_im in imgs.values():
        find_dependencies(labeled_im)

    if not already_saved:
        # If the image regions are not saved yet, use them to fill the image areas.
        for labeled_im in imgs.values():
            fill_image_areas(labeled_im)
    else:
        # Otherwise, load the image regions and assign them to the images.
        load_images(imgs)

    # Return images with their annotations.
    return imgs
