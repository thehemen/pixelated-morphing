import cv2
import numpy as np
from pixel_art.data import x_round
from pixel_art.bounding_box import *
from misc.misc import add_alpha_images
from misc.colors import get_color_transfer

classes = ['Head', 'Eye', 'Nose', 'Mouth', 'Hair', 'Helmet', 'Ear', 'Eyebrow']
special_classes = ['Hair', 'Helmet']

def get_weighted_mean(a, b, alpha):
    """Get a weighted mean of two values."""
    return int(round(a + (b - a) * alpha))

def to_first_half(value, half_width):
    """Move a value to the first half of the range."""
    value = half_width - (value - half_width)
    return value

def to_second_half(value, half_width):
    """Move a value to the second half of the range."""
    value = half_width + (half_width - value)
    return value

def add_images(img1, img2, alpha, width, height):
    """Add two images by resizing and blending them."""
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))

    img = cv2.addWeighted(img1, 1.0 - alpha, img2, alpha, 0.0)
    return img

class ImageMorphed:
    """
        ImageMorphed represents a morphed image.
    """
    def __init__(self, height, width, channels):
        """ImageMorphed constructor.

        Fields:
        img -- an image
        bboxes -- a dict of bounding boxes
        """
        self.img = np.zeros((height, width, channels), dtype=np.uint8)
        self.bboxes = {}

class MorphingState:
    """
        MorphingState represents a state of an image morphing.
    """
    def __init__(self, align_width, class_name, alpha, pair_index=None):
        """MorphingState constructor.
        Class fields are stored in the private 'values' dict.

        Fields:
        align_width -- a meta-pixel's alignment width
        class_name -- a class name
        alpha -- an alpha value of the image blending
        pair_index -- a pair index (if exists)
        """
        self.__values = {}
        self.__values['align_width'] = align_width
        self.__values['class_name'] = class_name
        self.__values['alpha'] = alpha
        self.__values['pair_index'] = pair_index

    def __getitem__(self, key):
        """Get a value by its name."""
        if key not in self.__values.keys():
            raise ValueError('This value name doesn\'t exist.')

        return self.__values[key]

    def add(self, alpha=None, pair_index=None):
        """Add a value and return the morphing state itself."""
        if pair_index is not None:
            self.__values['pair_index'] = pair_index

        if alpha is not None:
            self.__values['alpha'] = alpha

        return self

    def get_values(self):
        """Get a list of values."""
        return [self.__values[key] for key in self.__values.keys()]

class ImageMorphing:
    """
        ImageMorphing implements an image morphing.
    """
    def __init__(self, img1, img2, align_width):
        """ImageMorphing constructor.

        Fields:
        img1 -- the first image
        img2 -- the second image
        height -- a height value
        width -- a width value
        half_width -- a half width (used for paired regions)
        cell_size -- a width of the image region's cell (hair, helmet)
        align_width -- a meta-pixel's alignment width
        """
        self.img1 = img1
        self.img2 = img2

        self.height, self.width, self.channels = img1.image.shape
        self.half_width = int(round(self.width / 2.0))
        self.cell_size = int(round(self.width / 8.0))

        self.align_width = align_width

    def get_morphed_image(self, alpha=0.5, threshold=0.5):
        """Get a morphed image.

        Keyword arguments:
        alpha -- an alpha value of the image blending
        threshold -- a threshold value to keep the missing image areas
        """
        img = ImageMorphed(self.height, self.width, self.channels)

        for class_name in classes:
            morphingState = MorphingState(self.align_width, class_name, alpha)

            if self.__is_all(class_name) and class_name not in special_classes:
                img1, img2 = self.img1[class_name], self.img2[class_name]

                if not img1.is_paired:
                    img = self.__add_image(img, img1, img2, morphingState)
                else:
                    img = self.__add_image(img, img1.first, img2.first,
                        morphingState.add(pair_index=1))
                    img = self.__add_image(img, img1.second, img2.second,
                        morphingState.add(pair_index=2))

            elif self.__is_any(class_name):
                if alpha < threshold:
                    img_now = self.img1
                    head_img = self.img2['Head'].image_area
                else:
                    img_now = self.img2
                    head_img = self.img1['Head'].image_area

                alpha_upd = (threshold - alpha) * 2 if alpha < threshold else (alpha - threshold) * 2
                morphingState = morphingState.add(alpha=alpha_upd)

                if class_name not in img_now.get_classes():
                    continue

                im_region = img_now[class_name]
                head_bbox = img_now['Head'].bbox

                if not im_region.is_paired:
                    img = self.__add_image_with_gap(img, im_region, head_bbox, head_img, morphingState)
                else:
                    img = self.__add_image_with_gap(img, im_region.first, head_bbox, head_img,
                        morphingState.add(pair_index=1))
                    img = self.__add_image_with_gap(img, im_region.second, head_bbox, head_img,
                        morphingState.add(pair_index=2))

        return img.img

    def __add_image(self, img, region_1, region_2, morphingState):
        """Add two image regions.

        Keyword arguments:
        img -- a morphed image
        region_1 -- the first image region
        region_2 -- the second image region
        morphingState -- a current morphing state
        """
        # Get current morphing values.
        align_width, class_name, alpha, pair_index = morphingState.get_values()

        img1 = region_1.image_area
        img2 = region_2.image_area

        dependencies_1 = region_1.dependencies
        dependencies_2 = region_2.dependencies

        # Get x and y values by their weighted mean.
        x = self.__get_values(region_1.bbox, region_2.bbox, alpha, 'x', 'width')
        y = self.__get_values(region_1.bbox, region_2.bbox, alpha, 'y', 'height')

        # Get a bounding box by using center alignment.
        vals = {**x, **y}
        c_bbox = CenterBoundingBox(vals['x_center'], vals['y_center'], vals['width'], vals['height'])
        bbox = c_bbox.get_init_bbox()

        # Get dependencies common for both image regions.
        dependencies = self.__get_common_dependencies(dependencies_1, dependencies_2)

        if dependencies:
            # Adjust a bounding box by dependencies, if they exist.
            bbox = self.__adjust_by_dependencies(bbox, dependencies, img.bboxes)

            if pair_index:
                # Adjust a spacing width, if the image region is a paired one.
                bbox = self.__adjust_spacing_width(bbox, region_1.bbox, region_2.bbox, alpha, pair_index)

        # Align a bounding box by the meta-pixel's width.
        bbox = self.__align_by_value(bbox, align_width)

        # Add two images by the alpha value.
        width, height = bbox.width, bbox.height
        img_new = add_images(img1, img2, alpha, width, height)

        # Add an image region to the resulting image.
        x1, y1 = bbox.x1, bbox.y1
        img_area_new = add_alpha_images(img.img[y1: y1 + height, x1: x1 + width], img_new)
        img.img[y1: y1 + height, x1: x1 + width] = img_area_new

        # Define a class name.
        if not pair_index:
            class_name_now = class_name
        else:
            class_name_now = f'{class_name}_{pair_index}'

        # Add a bounding box to the morphed image.
        img.bboxes[class_name_now] = bbox

        # Return a morphed image.
        return img

    def __add_image_with_gap(self, img, im_region, head_bbox_init, head_img, morphingState):
        """Add two image regions when one is skipped.
        The first image's region is used, so the second image's region is skipped.

        Keyword arguments:
        img -- a morphed image
        im_region -- an image region
        head_bbox_init -- a bounding box of the first image's Head region
        head_img -- an image of the second image's Head region
        """
        # Get current morphing values.
        align_width, class_name, alpha, pair_index = morphingState.get_values()

        img_area = im_region.image_area
        bbox = im_region.bbox

        dependencies = im_region.dependencies
        head_bbox_upd = img.bboxes['Head']

        # Move a bounding box the same way as the morphed image's bounding box is moved.
        bbox = bbox.to_relative(head_bbox_init)
        bbox = bbox.to_absolute(head_bbox_upd)

        if dependencies:
            # Adjust a bounding box by dependencies, if they exist.
            bbox = self.__adjust_by_dependencies(bbox, dependencies, img.bboxes)

        if class_name == 'Hair':
            # If the class is Hair, crop it as defined.
            img_area, bbox = self.__get_hair_image(img_area, bbox, alpha)
        elif class_name == 'Helmet':
            # Otherwise, if the class is Helmet, crop it as defined too.
            img_area, bbox = self.__get_helmet_image(img_area, bbox, alpha)

        # If the height is zero, return to the image morphing.
        if img_area.shape[0] == 0:
            return img

        # Align a bounding box by the meta-pixel's width.
        bbox = self.__align_by_value(bbox, align_width)
        # Fix a bounding box shape by applying the size bounds.
        bbox = self.__apply_bounds(bbox)

        x1, y1 = bbox.x1, bbox.y1
        width, height = bbox.width, bbox.height

        # If width or height is zero, return to the image morphing.
        if np.any(np.array([width, height]) <= 0):
            return img

        img_new = cv2.resize(img_area, (width, height))
        head_img = cv2.resize(head_img, (width, height))

        # Apply a color transfer from the second image's Head region to this region.
        c_img_new = img_new.copy()
        c_img_new[:, :, :3] = get_color_transfer(head_img, c_img_new)

        # Add two images by the half of the alpha value.
        c_alpha = 1.0 - alpha * 2.0
        img_new = add_images(img_new, c_img_new, c_alpha, width, height)

        # Add an image region to the resulting image.
        img_area_new = add_alpha_images(img.img[y1: y1 + height, x1: x1 + width], img_new)
        img.img[y1: y1 + height, x1: x1 + width] = img_area_new

        # Define a class name.
        if not pair_index:
            class_name_now = class_name
        else:
            class_name_now = f'{class_name}_{pair_index}'

        # Add a bounding box to the morphed image.
        img.bboxes[class_name_now] = bbox

        # Return a morphed image.
        return img

    def __get_hair_image(self, img, bbox, alpha):
        """Get a hair image cropping it by height."""
        all_cell_num = int(bbox.height / self.cell_size)
        cell_num = int(round(all_cell_num * alpha))

        height = self.cell_size * cell_num
        bbox = BoundingBox(bbox.x1, bbox.y1, bbox.width, height)

        img = img[:height, :, :]
        return img, bbox

    def __get_helmet_image(self, img, bbox, alpha):
        """Get a helmet image cropping it by height."""
        helmet_skull = img[:self.half_width, :, :]
        helmet_ears = img[self.half_width:, :, :]

        all_cell_num = int(self.half_width / self.cell_size)
        cell_num = int(round(all_cell_num * alpha))

        height = self.cell_size * cell_num
        bbox = BoundingBox(bbox.x1, bbox.y1, bbox.width, height * 2)

        helmet_skull = helmet_skull[:height, :, :]
        helmet_ears = helmet_ears[:height, :, :]

        img = np.vstack([helmet_skull, helmet_ears])
        return img, bbox

    def __get_values(self, bbox_1, bbox_2, alpha, value, size, align_size=2):
        """Get values by their weighted mean."""
        value_center = get_weighted_mean(bbox_1[f'{value}_center'], bbox_2[f'{value}_center'], alpha)
        size_diff = bbox_2[size] - bbox_1[size]

        size_value = x_round(bbox_1[size] + size_diff * alpha, align_size)
        return {f'{value}_center': value_center, size: size_value}

    def __get_common_dependencies(self, dependencies_1, dependencies_2):
        """Get common dependencies by merging them."""
        dependencies = {}

        for dependency_1, value_pairs_1 in dependencies_1.items():
            for dependency_2, value_pairs_2 in dependencies_2.items():
                if dependency_1 == dependency_2:
                    for value_1_1, value_1_2 in value_pairs_1:
                        for value_2_1, value_2_2 in value_pairs_2:
                            if (value_1_1 == value_2_1) and (value_1_2 == value_2_2):
                                if dependency_1 not in dependencies.keys():
                                    dependencies[dependency_1] = []

                                dependencies[dependency_1].append(tuple((value_1_1, value_1_2)))

        return dependencies

    def __adjust_by_dependencies(self, bbox, dependencies, bboxes):
        """Adjust a bounding box by dependencies and other bounding boxes."""
        skipped_bbox = SkippedBoundingBox()

        for dependency_name, value_pairs in dependencies.items():
            for value_to, value_from in value_pairs:
                skipped_bbox[value_to] = bboxes[dependency_name][value_from]

        bbox = skipped_bbox.fill_missing_values(bbox, self.width)
        return bbox

    def __adjust_spacing_width(self, bbox, bbox_1, bbox_2, alpha, pair_index):
        """Adjust a spacing width between two regions."""
        if pair_index == 1:
            x2 = get_weighted_mean(bbox_1.x2, bbox_2.x2, alpha)
            bbox.width = x2 - bbox.x1
        else:
            x1 = get_weighted_mean(bbox_1.x1, bbox_2.x1, alpha)
            x2 = bbox.x2

            bbox.x1 = x1
            bbox.width = x2 - x1

        return bbox

    def __apply_bounds(self, bbox):
        """Apply width and height bounds to the bounding box."""
        bbox.x1 = max(0, bbox.x1)
        bbox.width = min(bbox.x2, self.width - 1) - bbox.x1

        bbox.y1 = max(0, bbox.y1)
        bbox.height = min(bbox.y2, self.height - 1) - bbox.y1
        return bbox

    def __align_by_value(self, bbox, align_width):
        """Align a bounding box by the meta pixel's width."""
        values = bbox.get_values()

        for key, value in values.items():
            # Check is a value in the second half of the image.
            is_reversed = value > self.half_width

            if is_reversed:
                # If so, move it to the first half.
                value = to_first_half(value, self.half_width)

            # Round a value by the meta-pixel's width.
            value = x_round(value, align_width)

            if is_reversed:
                # If so, move it to the second half.
                value = to_second_half(value, self.half_width)

            values[key] = value

        x1 = values['x1']
        y1 = values['y1']

        width = max(values['x2'] - x1, align_width)
        height = max(values['y2'] - y1, align_width)

        # Create a bounding box by the aligned values.
        bbox = BoundingBox(x1, y1, width, height)
        # Return an aligned bounding box.
        return bbox

    def __is_all(self, class_name):
        """Check if a class is present in all of images."""
        is_first = class_name in self.img1.get_classes()
        is_second = class_name in self.img2.get_classes()
        return is_first and is_second

    def __is_any(self, class_name):
        """Check if a class is present in any of images."""
        is_first = class_name in self.img1.get_classes()
        is_second = class_name in self.img2.get_classes()
        return is_first or is_second
