import numpy as np
from copy import deepcopy
from pixel_art.upscaling import apply_upscaling
from misc.misc import add_alpha_images
from misc.misc import unpixelate, pixelate
from misc.misc import apply_over_channels

classes = ['Head', 'Eye', 'Nose', 'Mouth', 'Hair', 'Helmet', 'Ear', 'Eyebrow']

def scale_image(img, width_from, width_to, is_upscaled=False):
    """Scale an image with the upscaling technique.

    Keyword arguments:
    img -- an image
    width_from -- a current width of the image
    width_to -- a required width of the image
    is_upscaled -- a flag that shows is the image upscaled or not
    """
    shape = np.array(img.shape)
    shape_from = (shape / width_from).round().astype(int)
    img = apply_over_channels(pixelate, img, shape_from)

    if is_upscaled:
        img_initial = img.copy()
        img_upscaled = apply_over_channels(unpixelate, img, 2)
        img = apply_upscaling(img_initial, img_upscaled)

    img = apply_over_channels(unpixelate, img, width_to)
    return img

class BaseRegion:
    """
        A base class for an image region.
    """
    def __init__(self, is_paired):
        """BaseRegion constructor.

        Fields:
        is_paired -- is an image region paired or not
        """
        self.is_paired = is_paired

class ImRegion(BaseRegion):
    """
        ImRegion represents an image region
        with the only image area.
    """
    def __init__(self, class_name, bbox, points):
        """ImRegion constructor.

        Fields:
        class_name -- a class name
        bbox -- a bounding box (left-top, absolute coordinates)
        points -- a list of points that represent a polygon
        image_area -- an image of the region
        dependencies -- a dict of value dependencies
        """
        self.class_name = class_name
        self.bbox = bbox
        self.points = points

        self.image_area = None
        self.dependencies = {}

        super().__init__(is_paired=False)

    def get_regions(self):
        """Get a list of a region itself."""
        return [self]

class PairRegion(BaseRegion):
    """
        PairRegion represents an image region
        with two image areas.
    """
    def __init__(self, class_name):
        """PairRegion constructor.

        Fields:
        class_name -- a class_name
        first -- the first image region
        second -- the second image region
        """
        self.class_name = class_name

        self.first = None
        self.second = None

        super().__init__(is_paired=True)

    def __getitem__(self, key):
        """Get a region by its index."""
        if key not in [0, 1]:
            raise ValueError('This region index doesn\'t exist.')

        if key == 0:
            return self.first
        elif key == 1:
            return self.second

    def add_region(self, region):
        """Add a region as the first or the second one."""
        if not self.first:
            self.first = region
        else:
            self.second = region

            # Swap regions by the x1 value.
            if self.first.bbox.x1 > self.second.bbox.x1:
                self.first, self.second = self.second, self.first

    def get_regions(self):
        """Get a list of two regions."""
        return [self.first, self.second]

class LabeledIm:
    """
        LabeledIm implements an image labeled by image regions.
    """
    min_level = 0
    max_level = 2

    def __init__(self, image_name, image, width):
        """LabeledIm constructor.

        Fields:
        image_name -- an image name
        image -- a complete RGB image
        regions -- a dict of regions by the class name
        width -- a current meta-pixel's width of the image
        width_initial -- an initial meta-pixel's width of the image
        image_saved -- a list of saved images
        regions_saved -- a list of saved regions
        level -- a level of the image upscaling
        """
        self.image_name = image_name
        self.image = image

        self.regions = {}

        self.width = width
        self.width_initial = width

        self.image_saved = []
        self.regions_saved = []

        self.level = 0

    def __getitem__(self, key):
        """Get a region by its class name."""
        if key not in self.regions.keys():
            raise ValueError('This class name doesn\'t exist.')

        return self.regions[key]

    def get_classes(self):
        """Get a list of class names."""
        return list(self.regions.keys())

    def add_region(self, region, is_pair_region):
        """Add a region or create the new one.

        Keyword arguments:
        region -- a region
        is_pair_region -- is a region paired or not
        """
        class_name = region.class_name

        if not is_pair_region:
            self.regions[class_name] = region
        else:
            if class_name not in self.regions.keys():
                self.regions[class_name] = PairRegion(class_name)

            self.regions[class_name].add_region(region)

    def get_image(self):
        """Get an image by blending its image areas."""
        img = np.zeros(self.get_shape(), dtype=np.uint8)

        for class_name in classes:
            if class_name not in self.regions.keys():
                continue

            for region in self.regions[class_name].get_regions():
                image_area = region.image_area
                bbox = region.bbox

                x1, y1 = bbox.x1, bbox.y1
                width, height = bbox.width, bbox.height

                image_area = add_alpha_images(img[y1: y1 + height, x1: x1 + width], image_area)
                img[y1: y1 + height, x1: x1 + width] = image_area

        return img

    def get_shape(self):
        """Get an image's shape."""
        return self.image.shape

    def upscale(self, num=1):
        """Upscale an image by the number of zooms in."""
        status = True

        for i in range(num):
            status &= self.__upscale()

        return status

    def downscale(self, num=1):
        """Downscale an image by the number of zooms out."""
        status = True

        for i in range(num):
            status &= self.__downscale()

        return status

    def __upscale(self):
        """Upscale an image."""
        # Skip if the scaling level is maximum.
        if self.level == self.max_level:
            return False

        if self.level == 0 and self.width == 4:
            # If level is 0 and width is 4, just zoom in the image.
            width_from = self.width
            self.width *= 2
            self.__scale(width_from, self.width)
        else:
            # Otherwise, upscale an image by the upscaling technique.
            self.image_saved.append(self.image.copy())
            self.regions_saved.append(deepcopy(self.regions))
            self.__scale(self.width, self.width * 2, width_modified=False)

        # Level up a scaling factor and highlight upscaling as successful.
        self.level += 1
        return True

    def __downscale(self):
        """Downscale an image."""
        # Skip if the scaling level is minimum.
        if self.level == self.min_level:
            return False

        if self.level == 1 and self.width_initial == 4:
            # If level is 1 and an initial width is 4, just zoom out the image.
            width_from = self.width
            self.width //= 2
            self.__scale(width_from, self.width)
        else:
            # Otherwise, downscale an image by returning its previous state.
            self.image = self.image_saved.pop()
            self.regions = self.regions_saved.pop()

        # Level down a scaling factor and highlight downscaling as successful.
        self.level -= 1
        return True

    def __scale(self, width_from, width_to, width_modified=True):
        """Scale an image by changing its meta-pixel's width."""
        # Scale a full image.
        self.image = scale_image(self.image, width_from, width_to)
        scale_coeff = width_to / width_from

        for class_name, region in self.regions.items():
            # If a width is not changed and a class name is either Hair or Head, upscale the image.
            if not width_modified and class_name in ['Hair', 'Head']:
                is_upscaled = True
            else:
                # Otherwise, just zoom in the image.
                is_upscaled = False

            # Get a list of regions.
            regions = region.get_regions()

            for img in regions:
                if not is_upscaled:
                    # Zoom in the image.
                    img.image_area = scale_image(img.image_area, width_from, width_to)
                else:
                    # Upscale the image.
                    img.image_area = scale_image(img.image_area, width_from, width_from, is_upscaled)

                # Scale the bounding box.
                img.bbox = img.bbox.get_scaled(scale_coeff)

        # Highlight scaling as successful.
        return True
