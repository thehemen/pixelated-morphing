import cv2
import glob
import tqdm
import shutil
import argparse
import numpy as np
from zipfile import ZipFile
from joblib import Parallel, delayed
from pixel_art.category_list import CategoryList
from pixel_art.morphing import ImageMorphing
from pixel_art.recoloring import ImageRecoloring
from pixel_art.style_transfer import StyleTransfer
from pixel_art.data import read_images_and_labels
from misc.misc import make_dir, get_random_beta
from misc.misc import pixelate, apply_over_channels

if __name__ == '__main__':
    """Generate images by the morphing algorithm."""
    parser = argparse.ArgumentParser(description='Generate new character images.')
    parser.add_argument('--n', type=int, default=100, help='the number of images')
    parser.add_argument('--recolor_value', type=float, default=0.9, help='the image recoloring probability')
    parser.add_argument('--change_style_value', type=float, default=0.9, help='the style changing probability')
    parser.add_argument('--alpha', type=float, default=5.0, help='the alpha value of beta distribution')
    parser.add_argument('--beta', type=float, default=5.0, help='the beta value of beta distribution')
    parser.add_argument('--width', type=int, default=4, help='the width to align')
    parser.add_argument('--upscale_num', type=int, default=2, help='the number of image zooms')
    parser.add_argument('--n_jobs', type=int, default=4, help='the number of processes run in parallel')
    parser.add_argument('--pixelate', default=False, action=argparse.BooleanOptionalAction, help='to pixelate an image by align width')
    args = parser.parse_args()

    tmp_dir_name = '.tmp'
    make_dir(tmp_dir_name)

    imgs = read_images_and_labels('pixelated-characters.json', already_saved=True)
    categoryList = CategoryList('categories.json')

    align_width = args.width * 2 ** args.upscale_num

    def get_character_name(characters_used, only_character_chosen=False):
        character_name = categoryList.get_character(characters_used, only_character_chosen)
        characters_used.append(character_name)
        return character_name

    def generate_image(index):
        characters_used = []

        img1 = imgs[get_character_name(characters_used)]
        img2 = imgs[get_character_name(characters_used)]
        img12 = [img1, img2]

        for j, img in enumerate(img12):
            if np.random.random() < args.recolor_value:
                img12[j] = ImageRecoloring(img).apply(hue=get_random_beta(-1.0, 1.0))

        for j, img in enumerate(img12):
            img.upscale(num=args.upscale_num)

            if np.random.random() < args.change_style_value:
                img_style_from = imgs[get_character_name(characters_used, only_character_chosen=True)]
                img12[j] = StyleTransfer(img, img_style_from, args.width).apply()

        img1, img2 = img12
        imageMorphing = ImageMorphing(img1, img2, align_width)

        alpha = get_random_beta(0.0, 1.0, args.alpha, args.beta)
        img = imageMorphing.get_morphed_image(alpha=alpha)

        if args.pixelate:
            shape = np.array(img.shape)
            shape = (shape / args.width).round().astype(int)
            img = apply_over_channels(pixelate, img, shape)

        cv2.imwrite(f'{tmp_dir_name}/{index}.png', img)

    Parallel(n_jobs=args.n_jobs)(delayed(generate_image)(i) for i in tqdm.tqdm(range(args.n)))

    with ZipFile('characters.zip', 'w') as myzip:
        for img_name in sorted(glob.glob(f'{tmp_dir_name}/*.png')):
            name_only = img_name.split('/')[-1]
            myzip.write(img_name, arcname=name_only)

    shutil.rmtree(tmp_dir_name)
