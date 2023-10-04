import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class AlignedValue:
    """
        AlignedValue represents a coordinate value (X or Y axis)
        that is aligned when the algorithm is done.
    """
    def __init__(self, index, value):
        """AlignedValue constructor.

        Fields:
        index -- an initial index of the value
        value -- a value by itself

        init_indices -- corresponding indices of left/right values
        init_value -- corresponding values located to the left/right from the value
        """
        self.index = index
        self.value = value

        self.init_indices = []
        self.init_values = []

    def add_init_value(self, index, value):
        """Add a value that is located to the left/right from this value."""
        self.init_indices.append(index)
        self.init_values.append(value)

class X_Align:
    """
        X_Align aligns an image grid values to the "raw" grid values
        over one coordinate axis that can be X or Y.
    """
    def __init__(self, im_values, raw_values, half_width):
        """X_Align constructor.

        Fields:
        half_width -- a half width of the image
        half_index -- a grid value index that corresponds to the half width (None if not presented)

        aligned_im_values -- aligned values of the image grid
        aligned_raw_values -- aligned values of the raw grid
        """
        self.half_width = half_width
        self.half_index = self.get_half_index(im_values, raw_values)

        self.aligned_im_values = self.get_aligned_values(im_values)
        self.aligned_raw_values = self.get_aligned_values(raw_values)

    def get_indices(self):
        """Get indices of the raw grid values corresponding to the image grid values."""
        im_values = np.array([x.value for x in self.aligned_im_values])
        raw_values = np.array([x.value for x in self.aligned_raw_values])

        # Calculate an euclidean distance matrix and match points by the Hungarian algorithm.
        row_dists = cdist(im_values[..., np.newaxis], raw_values[..., np.newaxis])
        aligned_indices = sorted(linear_sum_assignment(row_dists)[1])
        matched_raw_values = np.array(self.aligned_raw_values)[aligned_indices]

        init_im_indices = [x.init_indices for x in self.aligned_im_values]
        init_raw_indices = [x.init_indices for x in matched_raw_values]

        init_im_indices = np.array(init_im_indices).ravel()
        init_raw_indices = np.array(init_raw_indices).ravel()

        # Get a list of corresponding image and raw indices and sort it by image indices.
        init_indices = [tuple((x, y)) for x, y in zip(init_im_indices, init_raw_indices)]
        init_indices.sort(key=lambda x: x[0])

        # Get a raw grid indices only.
        indices = [x[1] for x in init_indices]

        # If a half index exists, insert it to the half of the index list.
        if self.half_index:
            indices.insert(len(indices) // 2, self.half_index)

        # Return a list of desired indices.
        return indices

    def get_half_index(self, im_values, raw_values):
        """If an image grid value number is odd, return the half index of the raw grid values."""
        if len(im_values) % 2 == 1:
            return len(raw_values) // 2
        else:
            return None

    def get_aligned_values(self, values):
        """Get values aligned by the half width of the image."""
        aligned_values = []

        val_num = len(values)
        half_val_num = val_num // 2

        left_index = half_val_num
        right_index = half_val_num

        if val_num % 2 == 1:
            right_index += 1

        for left_value, right_value in zip(reversed(values[:left_index]), values[right_index:]):
            aligned_left_value = self.half_width - left_value
            aligned_right_value = right_value - self.half_width

            aligned_index = len(aligned_values)
            mean_value = (aligned_left_value + aligned_right_value) / 2.0
            aligned_value = AlignedValue(aligned_index, mean_value)

            init_left_index = left_index - aligned_index - 1
            init_right_index = right_index + aligned_index

            aligned_value.add_init_value(init_left_index, left_value)
            aligned_value.add_init_value(init_right_index, right_value)
            aligned_values.append(aligned_value)

        return aligned_values
