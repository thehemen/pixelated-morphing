import cv2
import tqdm
import glob
from misc.misc import make_dir
from pixel_art.im_align import Im_Align
from pixel_art.labeled_im import LabeledIm

if __name__ == '__main__':
    """Add images aligned by their meta-pixel's width."""
    out_dir_name = 'aligned_assets/'
    make_dir(out_dir_name)

    for img_path in tqdm.tqdm(glob.glob('assets/*.png')):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_name = img_path.split('/')[-1]

        im_align = Im_Align(img)
        img = im_align.transform()

        img_name_no_ext = img_name.split('.')[0]
        labeled_im = LabeledIm(img_name_no_ext, img, im_align.cell_width)

        labeled_im.upscale()
        labeled_im.downscale()

        aligned_img = labeled_im.image
        cv2.imwrite(f'{out_dir_name}{img_name}', aligned_img)
