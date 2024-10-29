import glob
import shutil
import argparse
import tqdm
from zipfile import ZipFile
from joblib import Parallel, delayed
from pixel_art.data import read_images_and_labels
from pixel_art.category_list import CategoryList
from pixel_art.im_generation import ImGeneration
from misc.misc import make_dir

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
    parser.add_argument('--rect', default=False, action=argparse.BooleanOptionalAction, help='to use only rectangular images')
    parser.add_argument('--pixelate', default=False, action=argparse.BooleanOptionalAction, help='to pixelate an image by align width')
    args = parser.parse_args()

    tmp_dir_name = '.tmp'
    make_dir(tmp_dir_name)

    imgs = read_images_and_labels('pixelated-characters.json', already_saved=True)
    categoryList = CategoryList('categories.json', args.rect)

    paramDict = {
        'recolor_value': args.recolor_value,
        'change_style_value': args.change_style_value,
        'alpha': args.alpha,
        'beta': args.beta,
        'width': args.width,
        'upscale_num': args.upscale_num,
        'pixelate': args.pixelate
    }

    imGeneration = ImGeneration(imgs, categoryList, paramDict)

    Parallel(n_jobs=args.n_jobs)(delayed(imGeneration.generate_image)(f'{tmp_dir_name}/{i}.png') for i in tqdm.tqdm(range(args.n)))

    with ZipFile('characters.zip', 'w') as myzip:
        for img_name in sorted(glob.glob(f'{tmp_dir_name}/*.png')):
            name_only = img_name.split('/')[-1]
            myzip.write(img_name, arcname=name_only)

    shutil.rmtree(tmp_dir_name)
