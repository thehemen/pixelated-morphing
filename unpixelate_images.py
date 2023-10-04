import os
import cv2
import glob
import tqdm
import argparse
from misc.misc import make_dir
from misc.misc import unpixelate, apply_over_channels

if __name__ == '__main__':
    """Unpixelate (uncompress) characters images if they were pixelated (compressed) earlier."""
    parser = argparse.ArgumentParser(description='Unpixelate (upscale) new character images.')
    parser.add_argument('--input', default='../images/', help='The initial images folder.')
    parser.add_argument('--output', default='../images_updated/', help='The unpixelated images folder.')
    parser.add_argument('--width', type=int, default=4, help='the width to align')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise Exception('The initial images folder doesn\'t exist.')

    make_dir(args.output)

    for image_name in tqdm.tqdm(sorted(glob.glob(f'{args.input}*.png'))):
        image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        image = apply_over_channels(unpixelate, image, args.width)
        name_only = image_name.split('/')[-1]
        cv2.imwrite(f'{args.output}{name_only}', image)
