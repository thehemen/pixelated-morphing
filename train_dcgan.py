import os
import cv2
import glob
import tqdm
import random
import argparse
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import get_ema_multi_avg_fn, update_bn
from copy import copy
from torchinfo import summary
from albumentations.pytorch import ToTensorV2
from pixel_art.data import read_images_and_labels
from pixel_art.category_list import CategoryList
from pixel_art.im_generation import ImGeneration
from misc.misc import make_dir
from dcgan.model import weights_init
from dcgan.model import Generator, Discriminator
from dcgan.visualization import plot_images, plot_loss_history
from dcgan.visualization import plot_image_comparison, save_output
from dcgan.image_history import ImageHistory
from dcgan.pixel_dataset import G_generator, D_real_generator
from dcgan.pixel_dataset import D_fake_generator, D_generator
from dcgan.pixel_dataset import PixelDataset

if __name__ == '__main__':
    """Train a DCGAN model."""
    parser = argparse.ArgumentParser(description='Train the DCGAN image generation model.')
    parser.add_argument('--n', type=int, default=1024, help='the number of images')
    parser.add_argument('--seed', type=int, default=42, help='the seed value')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of the batch')
    parser.add_argument('--num_epochs', type=int, default=25, help='the number of epochs')
    parser.add_argument('--checkpoint_epoch', type=int, default=5, help='the number of epochs when the model is saved')
    parser.add_argument('--img_epoch', type=int, default=1, help='the number of epochs when the image is saved')
    parser.add_argument('--change_image_num', type=int, default=5, help='the number of image history epochs')
    parser.add_argument('--change_image_value', type=float, default=0.25, help='the probability of image change')
    parser.add_argument('--noise_value', type=float, default=0.1, help='the max value of noise added to labels')
    parser.add_argument('--workers', type=int, default=4, help='the number of processes run in parallel')
    parser.add_argument('--gpu', default=True, action=argparse.BooleanOptionalAction, help='to use the GPU device')
    args = parser.parse_args()

    print(f'Random Seed: {args.seed}.')
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    make_dir('output/')

    make_dir('output/images/')
    make_dir('output/checkpoints/')

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    imgs = read_images_and_labels('pixelated-characters.json', already_saved=True)

    paramDict = {
        'recolor_value': 0.9,       # the image recoloring probability
        'change_style_value': 0.9,  # the style changing probability
        'alpha': 5.0,               # the alpha value of beta distribution
        'beta': 5.0,                # the beta value of beta distribution
        'width': 4,                 # the width to align
        'upscale_num': 0,           # the number of image zooms
        'rect': True,               # to use only rectangular images
        'pixelate': False           # to pixelate an image by align width
    }

    categoryList = CategoryList('categories.json', paramDict['rect'])
    imGeneration = ImGeneration(imgs, categoryList, paramDict)

    imageHistory = ImageHistory(args.change_image_num)

    dataset = PixelDataset(args.n, args.workers, imGeneration, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    device = torch.device('cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    real_batch = next(iter(dataloader))
    plot_images(real_batch, device, n=64)

    nc = 4
    nz = 300
    ngf = 64
    ndf = 64
    ema_decay = 0.999

    netG = Generator(nz, ngf, nc).to(device)
    netG.apply(weights_init)

    print('Generator.')
    summary(netG, input_size=(args.batch_size, nz, 1, 1))

    netD = Discriminator(nc, ndf).to(device)
    netD.apply(weights_init)

    print('Discriminator.')
    summary(netD, input_size=(args.batch_size, nc, ngf, ngf))

    ema_netG = AveragedModel(netG, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    ema_netD = AveragedModel(netD, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(ngf, nz, 1, 1, device=device)

    real_label = 1.0
    fake_label = 0.0

    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0003, betas=(0.5, 0.999))

    G_losses = []
    D_losses = []

    with tqdm.tqdm(total=args.num_epochs, position=0, leave=True) as tqdm_bar:
        for epoch in range(args.num_epochs):
            G_loss_list = []
            D_loss_list = []

            imageHistory.start_epoch(epoch)

            for i, data in enumerate(dataloader, 0):
                # Train the discriminator.
                netD.zero_grad()

                # Train with all-real batch.
                b_size = data.size(0)
                data = data.to(device)

                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                label.uniform_(real_label - args.noise_value, real_label)

                output = netD(data).view(-1)
                errD_real = criterion(output, label)

                # Train with all-fake batch.
                noise = torch.randn(b_size, nz, 1, 1, device=device)

                fake = netG(noise)
                label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
                label.uniform_(fake_label, fake_label + args.noise_value)

                if not i or np.random.random() < args.change_image_value:
                    imageHistory.add_images(epoch, fake.detach())

                if epoch and np.random.random() < args.change_image_value:
                    fake = imageHistory.get_images(b_size)

                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)

                errD = errD_real + errD_fake
                errD.backward()

                optimizerD.step()
                ema_netD.update_parameters(netD)

                # Train the generator.
                netG.zero_grad()
                label.uniform_(real_label - args.noise_value, real_label)

                output = netD(fake).view(-1)
                errG = criterion(output, label)

                errG.backward()

                optimizerG.step()
                ema_netG.update_parameters(netG)

                G_loss_list.append(errG.item())
                D_loss_list.append(errD.item())

                if i == len(dataloader) - 1 and (epoch + 1) % args.img_epoch == 0:
                    save_output(epoch, netG, fixed_noise)

            if (epoch + 1) % args.checkpoint_epoch == 0:
                torch.save(netG.state_dict(), f'output/checkpoints/netG_{epoch + 1}.pt')
                torch.save(netD.state_dict(), f'output/checkpoints/netD_{epoch + 1}.pt')

            imageHistory.stop_epoch(epoch)

            G_loss = np.mean(G_loss_list)
            D_loss = np.mean(D_loss_list)

            G_losses.append(G_loss)
            D_losses.append(D_loss)

            tqdm_bar.update(1)
            tqdm_bar.set_description(f'Loss_D: {D_loss:.4f} Loss_G: {G_loss:.4f}')

    g_generator = G_generator(dataloader, nz, device)

    d_real_generator = D_real_generator(dataloader, device)
    d_fake_generator = D_fake_generator(dataloader, netG, nz, device)

    d_generator = D_generator(d_real_generator, d_fake_generator)

    update_bn(g_generator, ema_netG)
    update_bn(d_generator, ema_netD)

    img = save_output(epoch, ema_netG, fixed_noise)

    plot_loss_history(G_losses, D_losses)

    real_batch = next(iter(dataloader))
    plot_image_comparison(img, real_batch, device, n=64)

    torch.save(ema_netG.state_dict(), 'output/netG.pt')
    print('Generator saved to output/netG.pt.')

    torch.save(ema_netD.state_dict(), 'output/netD.pt')
    print('Discriminator saved to output/netD.pt.')
