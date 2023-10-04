import numpy as np

class BoundingBox:
    """
        BoundingBox implements a bounding box,
        a rectangle field on the image.

        Starting Point: Left, Top.
        BBox Orientation: Absolute.
    """
    def __init__(self, x1, y1, width, height):
        """BoundingBox constructor.

        Fields:
        x1 -- x1 value
        y1 -- y1 value
        width -- width value
        height -- height value
        """
        self.x1 = x1
        self.y1 = y1

        self.width = width
        self.height = height

    def to_relative(self, bbox):
        """Get a relative bounding box."""
        x1 = (self.x1 - bbox.x1) / bbox.width
        y1 = (self.y1 - bbox.y1) / bbox.height

        width = self.width / bbox.width
        height = self.height / bbox.height
        return RelativeBoundingBox(x1, y1, width, height)

    def get_center_bbox(self):
        """Get a center bounding box."""
        return CenterBoundingBox(self.x_center, self.y_center, self.width, self.height)

    def get_values(self):
        """Get a dictionary with the bounding box values."""
        values = {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}
        return values

    def get_scaled(self, coeff):
        """Get a bounding box scaled by coeff."""
        x1 = int(round(self.x1 * coeff))
        y1 = int(round(self.y1 * coeff))

        width = int(round(self.width * coeff))
        height = int(round(self.height * coeff))
        return BoundingBox(x1, y1, width, height)

    @property
    def x2(self):
        """Calculate the x2 value."""
        return self.x1 + self.width

    @property
    def y2(self):
        """Calculate the y2 value."""
        return self.y1 + self.height

    @property
    def x_center(self):
        """Calculate the x_center value."""
        return int(round(np.mean([self.x1, self.x2])))

    @property
    def y_center(self):
        """Calculate the y_center value."""
        return int(round(np.mean([self.y1, self.y2])))

    def __getitem__(self, key):
        """Get a value by key."""
        if key not in ['x1', 'y1', 'x2', 'y2', 'x_center', 'y_center', 'width', 'height']:
            raise ValueError('This value name doesn\'t exist.')

        if key == 'x1':
            return self.x1
        elif key =='y1':
            return self.y1
        elif key == 'x2':
            return self.x2
        elif key == 'y2':
            return self.y2
        elif key == 'x_center':
            return self.x_center
        elif key == 'y_center':
            return self.y_center
        elif key == 'width':
            return self.width
        elif key == 'height':
            return self.height

class CenterBoundingBox:
    """
        CenterBoundingBox implements a center version
        of the bounding box.

        Starting Point: X-Center, Y-Center.
        BBox Orientation: Absolute.
    """
    def __init__(self, x_center, y_center, width, height):
        """CenterBoundingBox constructor.

        Fields:
        x_center -- x_center value
        y_center -- y_center value
        width -- width value
        height -- height value
        """
        self.x_center = x_center
        self.y_center = y_center

        self.width = width
        self.height = height

    def get_init_bbox(self):
        """Get an initial (with a left-top point) bounding box."""
        return BoundingBox(self.x1, self.y1, self.width, self.height)

    @property
    def x1(self):
        """Calculate the x1 value."""
        return int(round(self.x_center - self.width / 2.0))

    @property
    def y1(self):
        """Calculate the y1 value."""
        return int(round(self.y_center - self.height / 2.0))

    @property
    def x2(self):
        """Calculate the x2 value."""
        return int(round(self.x_center + self.width / 2.0))

    @property
    def y2(self):
        """Calculate the y2 value."""
        return int(round(self.y_center + self.height / 2.0))

class SkippedBoundingBox:
    """
        SkippedBoundingBox implements a skipped bounding box
        that has missing values.
    """
    def __init__(self):
        """SkippedBoundingBox constructor.

        Fields:
        values -- a private dict of values
        """
        self.__values = {}

    def __setitem__(self, key, value):
        """Assign a value by key."""
        self.__values[key] = value

    def fill_missing_values(self, bbox, max_size):
        """Fill missing values by another bounding box and max size."""
        for value, size in [['x', 'width'], ['y', 'height']]:
            self.__fill_missing_values(value, size, bbox, max_size)

        x1, y1 = self.__values['x1'], self.__values['y1']
        width, height = self.__values['x2'] - x1, self.__values['y2'] - y1
        return BoundingBox(x1, y1, width, height)

    def __fill_missing_values(self, value, size, bbox, max_size):
        """Fill missing values for a specific pair of coordinates.

        Keyword arguments:
        value -- a name of the value (x[1], y[1])
        size -- a name of the size (width, height)
        bbox -- a bounding box from which it needs to get values
        max_size -- a maximum size of a bounding box (width, height)
        """
        first = f'{value}1'
        second = f'{value}2'

        is_first = first in self.__values.keys()
        is_second = second in self.__values.keys()

        if is_first and not is_second:
            self.__values[second] = min(self.__values[first] + bbox[size], max_size)
        elif not is_first and is_second:
            self.__values[first] = max(0, self.__values[second] - bbox[size])
        elif not is_first and not is_second:
            self.__values[first] = bbox[first]
            self.__values[second] = bbox[second]

class RelativeBoundingBox:
    """
        RelativeBoundingBox implements a bounding box
        that is relative to another one.
    """
    def __init__(self, x1, y1, width, height):
        """RelativeBoundingBox constructor.

        Fields:
        x1 -- x1 value
        y1 -- y1 value
        width -- width value
        height -- height value
        """
        self.x1 = x1
        self.y1 = y1

        self.width = width
        self.height = height

    def to_absolute(self, bbox):
        """Get an absolute bounding box."""
        x1 = int(round(self.x1 * bbox.width + bbox.x1))
        y1 = int(round(self.y1 * bbox.height + bbox.y1))

        width = int(round(self.width * bbox.width))
        height = int(round(self.height * bbox.height))
        return BoundingBox(x1, y1, width, height)
