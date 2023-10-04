import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label
from pixel_art.x_align import X_Align

"""

Images can have different number of cells.
A 64x64 image can be represented
as a 8x8 or 16x16 grid of cells.
So, each cell in an image grid has size of 8x8 or 4x4.
An ASCII-representation is provided here.

---------------------          ---------------------
|    |    |    |    |          |         |         |
|    |    |    |    |          |         |         |
---------------------          |         |         |
|    |    |    |    |          |         |         |
|    |    |    |    |          |         |         |
---------------------    OR    ---------------------
|    |    |    |    |          |         |         |
|    |    |    |    |          |         |         |
---------------------          |         |         |
|    |    |    |    |          |         |         |
|    |    |    |    |          |         |         |
---------------------          ---------------------

The structure of an image grid can be inconsistent,
so it makes an image morphing process a bit complicated.
To make the morphing process easier and the quality
of the given images better, an image alignment is used here.

"""

def delta_E(im1, im2):
    """Get a color difference between two images, represented in the CIELAB model."""
    return np.sqrt(np.sum((im1 - im2) ** 2, axis=-1)) / 255

def get_hls_color():
    """Get a random color from the HLS color space."""
    hue = np.random.randint(0, 180)
    lightness = np.random.randint(100, 150)
    saturation = np.random.randint(128, 256)
    return hue, lightness, saturation

def get_labeled_value(x):
    """Label connected components in the color difference array."""
    kernel = np.ones((3,), dtype=int)
    labeled, ncomponents = label(x, kernel)
    return labeled

def get_value_diff(x):
    """Get a value of the color difference by the values of connected components."""
    segment_scores = {i: sum(x == i) for i in range(1, max(x) + 1)}
    diff_sum = sum(segment_scores.values())
    diff = sum([num * (num / diff_sum) for num in segment_scores.values()])
    return diff

class Grid:
    """
        A base class for a grid of cells.
        It cen be an image grid or a raw grid.
    """
    def __init__(self, x, y, shape):
        """Grid constructor.

        Fields:
        x -- a list of grid values along the x-axis
        y -- a list of grid values along the y-axis
        shape -- a shape of the grid
        complexity -- a complexity of the grid
        """
        self.x = x
        self.y = y
        self.shape = shape
        self.complexity = max(len(self.x), len(self.y))

    def draw_grid_map(self):
        """Draw a grid map by grid lines."""
        grid_map = np.zeros((*self.shape, 3), dtype=np.uint8)

        color = get_hls_color()
        grid_map[self.y, :, :] = color
        grid_map[:, self.x, :] = color

        grid_map = cv2.cvtColor(grid_map, cv2.COLOR_HLS2BGR)
        return grid_map

    def __getitem__(self, key):
        """Get x or y grid values by the key."""
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y
        else:
            return None

class RawGrid(Grid):
    """
        RawGrid represents an initial 'raw' grid
        that is used as the target for an image alignment.
    """
    def __init__(self, cell_width, im_width):
        """RawGrid constructor.

        Keyword arguments:
        cell_width -- a width of the cell
        im_width -- a width of the image
        """
        x = np.array(range(cell_width, im_width, cell_width))
        y = np.array(range(cell_width, im_width, cell_width))
        super().__init__(x, y, (im_width, im_width))

class ImGrid(Grid):
    """
        ImGrid represents an image grid
        that is aligned by the raw grid.
    """
    def __init__(self, im):
        """ImGrid constructor.

        Fields:
        im -- a RGB image (np.uint8 format)
        diff -- a difference map of the image
        im_width -- a width of the image
        half_width -- a half width of the image
        """
        self.im = im
        self.diff = self.get_difference_map()

        self.im_width = self.im.shape[0]
        self.half_width = self.im_width // 2

        x, y = self.get_grid_param()
        super().__init__(x, y, self.im.shape[:2])

    def get_difference_map(self, min_diff=0.02):
        """Get a difference map of the image.

        Keyword arguments:
        min_diff -- a minimum difference between pixels to apply line thinning.
        """
        # Convert to Lab color space.
        lab = cv2.cvtColor(self.im.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        h, w, c = lab.shape

        # Get delta E between pixels.
        diff = delta_E(lab[:h - 1, :w - 1], lab[1:, 1:])
        diff = np.pad(diff, ((1, 0), (1, 0)))

        # Convert to uint8 format to apply line thinning.
        diff[diff > min_diff] = 1.0
        diff = (diff * 255).astype(np.uint8)
        diff = cv2.ximgproc.thinning(diff)

        # Convert back to float format.
        diff = (diff / 255).astype(float)
        return diff

    def remove_close_elements(self, arr, min_length=2):
        """Remove close elements in an array.

        Keyword arguments:
        arr -- an array of values
        min_length -- a minimum length between values
        """
        if len(arr) < 2:
            return arr

        diff = np.empty(arr.shape)
        diff[0] = np.inf
        diff[1:] = np.diff(arr)
        mask = diff > min_length
        new_arr = arr[mask]
        return new_arr

    def remove_close_elements_by_x(self, arr, min_length=1):
        """Remove close elements in an array while taking into account
        the symmetry of values along the x-axis.

        Keyword arguments:
        arr -- an array of values
        min_length -- a minimum length between the value and the half width
        """
        half_width = self.half_width

        symm_arr = np.array([x if x < half_width else half_width - (x - half_width) for x in arr])
        new_symm_arr = self.remove_close_elements(symm_arr)
        new_arr = []

        for x in new_symm_arr:
            new_arr.append(x)

            if abs(x - half_width) > min_length:
                new_arr.append(2 * half_width - x)

        new_arr.sort()
        new_arr = np.array(new_arr)
        return new_arr

    def get_grid_param(self, min_diff=3.0):
        """Get params of the grid along x and y axes.

        Keyword arguments:
        min_diff -- a minimum difference between grid values
        """
        x_labels = np.apply_along_axis(get_labeled_value, 0, self.diff)
        y_labels = np.apply_along_axis(get_labeled_value, 1, self.diff)

        x_diff = np.apply_along_axis(get_value_diff, 0, x_labels)
        y_diff = np.apply_along_axis(get_value_diff, 1, y_labels)

        x_grid = np.nonzero(x_diff > min_diff)[0]
        y_grid = np.nonzero(y_diff > min_diff)[0]

        x_grid = self.remove_close_elements_by_x(x_grid)
        y_grid = self.remove_close_elements(y_grid)

        x_grid = np.array([x for x in x_grid if x not in [0, self.im_width - 1]])
        y_grid = np.array([x for x in y_grid if x not in [0, self.im_width - 1]])
        return x_grid, y_grid

    def transform(self, mean_lines):
        """Transforms a RGB image by mean lines.

        Keyword arguments:
        mean_lines -- mean values of grid lines
        """
        height, width, channels = self.im.shape
        im_t = self.im.copy()
        im_t[:, :, :] = 0

        # Initialize x and y lists by borders of the image.
        xs = [0, width]
        ys = [0, height]

        # Add mean values to the specified lists.
        xs[1: 1] = mean_lines['x']
        ys[1: 1] = mean_lines['y']

        def get_mean_value(x, axis):
            """
                Get a mean value that matches the specified value,
                if exists. Otherwise, return a value itself.
            """
            if x in mean_lines[axis].keys():
                return mean_lines[axis][x]
            else:
                return x

        # Iterate through neighbouring grid lines.
        for x1, x2 in zip(xs, xs[1:]):
            m_x1 = get_mean_value(x1, 'x')
            m_x2 = get_mean_value(x2, 'x')

            for y1, y2 in zip(ys, ys[1:]):
                m_y1 = get_mean_value(y1, 'y')
                m_y2 = get_mean_value(y2, 'y')

                # Resize each cell of the image.
                img_area = self.im[y1: y2, x1: x2]
                img_area = cv2.resize(img_area, (m_x2 - m_x1, m_y2 - m_y1))
                im_t[m_y1: m_y2, m_x1: m_x2] = img_area

        # Return an aligned image.
        return im_t

class Im_Align:
    """
        Im_Align aligns an image by the grid
        while taking into account the complexity of the image.
    """
    def __init__(self, img, min_cell_width=4, max_cell_width=8, im_width=64):
        """Im_Align constructor.

        Fields:
        grid -- a list of ImGrid and RawGrid class instances
        cell_width -- a width of the grid cell
        im_width -- a width of the image
        half_width -- a half width of the image
        """
        self.grid = [ImGrid(img)]

        self.cell_width = max_cell_width if self.grid[0].complexity < max_cell_width else min_cell_width
        self.grid.append(RawGrid(self.cell_width, im_width))

        self.im_width = im_width
        self.half_width = im_width // 2

    def get_mean_by_axis(self, axis):
        """Get mean values by the axis.

        Keyword arguments:
        axis -- an axis of the image or raw grid
        """
        mean_values = {}
        im_values, raw_values = [grid[axis] for grid in self.grid]

        if axis == 'x':
            x_align = X_Align(im_values, raw_values, self.half_width)
            indices = x_align.get_indices()
        else:
            value_dists = cdist(im_values[..., np.newaxis], raw_values[..., np.newaxis])
            indices = sorted(linear_sum_assignment(value_dists)[1])

        for first, second in zip(im_values, raw_values[indices]):
            mean_values[first] = second

        return mean_values

    def transform(self):
        """Transforms an image by mean lines."""
        mean_lines = {}
        mean_lines['x'] = self.get_mean_by_axis(axis='x')
        mean_lines['y'] = self.get_mean_by_axis(axis='y')

        img = self.grid[0].transform(mean_lines)
        return img
