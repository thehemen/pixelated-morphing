import cv2
import torch
import random
import argparse
import numpy as np
import torchvision.utils as vutils
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import get_ema_multi_avg_fn
from dcgan.model import Generator
from dcgan.visualization import get_opencv_image
from misc.misc import unpixelate, apply_over_channels

def random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    """Test a DCGAN model."""
    parser = argparse.ArgumentParser(description='Test the DCGAN model.')
    parser.add_argument('--model_name', default='weights/netG.pt', help='the filename of the model')
    parser.add_argument('--seed_first', type=int, default=28, help='the first image seed value')
    parser.add_argument('--seed_second', type=int, default=35, help='the second image seed value')
    parser.add_argument('--width', type=int, default=4, help='the width to align')
    parser.add_argument('--frame_num', type=int, default=100, help='the number of frames')
    parser.add_argument('--delay', type=int, default=10, help='the delay between frames in ms')
    parser.add_argument('--gpu', default=True, action=argparse.BooleanOptionalAction, help='to use the GPU device')
    parser.add_argument('--image', default=True, action=argparse.BooleanOptionalAction, help='to use the single image')
    args = parser.parse_args()

    seed_first = int(str(args.seed_first)[::-1])
    random_seed(seed_first)

    device = torch.device('cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    nc = 4
    nz = 300
    ngf = 64
    ema_decay = 0.999

    netG_base = Generator(nz, ngf, nc).to(device)
    netG = AveragedModel(netG_base, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    netG.load_state_dict(torch.load(args.model_name, weights_only=True))
    netG.eval()

    seed_second = int(str(args.seed_second)[::-1])

    fixed_noise = torch.randn(1, nz, 1, 1, device=device)

    if args.image:
        with torch.no_grad():
            output = netG(fixed_noise).detach().cpu()[0]

        img = get_opencv_image(output)
        img = apply_over_channels(unpixelate, img, args.width)
        cv2.imshow('Image', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        i = 0
        eps = 1.0 / args.frame_num

        random_seed(seed_second)

        updated_noise = torch.randn(1, nz, 1, 1, device=device)
        added_noise = eps * (updated_noise - fixed_noise)

        for i in range(args.frame_num):
            fixed_noise[:, :, :, :] += added_noise

            with torch.no_grad():
                output = netG(fixed_noise).detach().cpu()[0]

            img = get_opencv_image(output)
            img = apply_over_channels(unpixelate, img, args.width)
            cv2.imshow('Image', img)
            cv2.waitKey(args.delay)

        cv2.destroyAllWindows()

    print(f'Random Seed: {args.seed_first}.')
