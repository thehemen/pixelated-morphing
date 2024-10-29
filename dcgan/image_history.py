import torch
import numpy as np

class ImageHistory:
    """
        ImageHistory stores the history of fake images
        that is used to let the discriminator converge.
    """
    def __init__(self, epoch_num):
        """ImageHistory constructor.

        Fields:
        epoch_num -- a number of epochs
        image_history -- a dict to stores images by epochs
        """
        self.epoch_num = epoch_num
        self.image_history = {}

    def start_epoch(self, epoch):
        """Start epoch by creation of new history epoch."""
        if len(self.image_history) == self.epoch_num:
            min_epoch = min(self.image_history.keys())
            self.image_history.pop(min_epoch)

        self.image_history[epoch] = []

    def stop_epoch(self, epoch):
        """Stop epoch by conversion of the image list to the pytorch tensor."""
        self.image_history[epoch] = torch.cat(self.image_history[epoch])

    def add_images(self, epoch, images):
        """Add images to the image history."""
        self.image_history[epoch].append(images.clone())

    def get_images(self, num):
        """Get a random list of images by the random epoch."""
        min_epoch = min(self.image_history.keys())
        max_epoch = max(self.image_history.keys())

        epoch = np.random.randint(min_epoch, max_epoch)
        epoch_images = self.image_history[epoch]

        weights = torch.ones(len(epoch_images))
        indices = weights.multinomial(num, replacement=False)

        images = epoch_images[indices]
        return images
