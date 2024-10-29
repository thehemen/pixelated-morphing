import cv2
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import Dataset

def G_generator(dataloader, nz, device):
    """Generate noise vectors for the generator."""
    for i, data in enumerate(dataloader, 0):
        yield torch.randn(data.size(0), nz, 1, 1, device=device)

def D_real_generator(dataloader, device):
    """Generate real images for the discriminator."""
    for i, data in enumerate(dataloader, 0):
        yield data.to(device)

def D_fake_generator(dataloader, netG, nz, device):
    """Generate fake images for the discriminator."""
    for i, data in enumerate(dataloader, 0):
        yield netG(torch.randn(data.size(0), nz, 1, 1, device=device).detach())

def D_generator(d_real_generator, d_fake_generator):
    """Generate real and fake images for the discriminator."""
    for data in zip(d_real_generator, d_fake_generator):
        for sample in data:
            yield sample

class PixelDataset(Dataset):
    """
        PixelDataset generates images
        to be used in the dataloader.
    """
    def __init__(self, n, workers, imGeneration, transform=None):
        """PixelDataset constructor.

        Fields:
        n -- a number of images
        workers -- a number of workers
        imGeneration -- an imGeneration class instance
        transform -- a list of transforms applied to the image
        """
        self.n = n
        self.workers = workers
        self.imGeneration = imGeneration
        self.transform = transform

        with Parallel(n_jobs=self.workers) as parallel:
            self.imgs = parallel(delayed(imGeneration.generate_image)(color_space='rgb') for i in tqdm(range(self.n)))

    def __len__(self):
        """Get a length of the dataset."""
        return self.n

    def __getitem__(self, idx):
        """Get an image by the index."""
        image = self.imgs[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image
