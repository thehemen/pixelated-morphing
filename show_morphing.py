import cv2
import argparse
from pixel_art.morphing import ImageMorphing
from pixel_art.recoloring import ImageRecoloring
from pixel_art.style_transfer import StyleTransfer
from pixel_art.data import read_images_and_labels
from misc.misc import get_random_beta

if __name__ == '__main__':
    """Show an image morphing process with recoloring and swap style options."""
    parser = argparse.ArgumentParser(description='Show two characters morphing result')
    parser.add_argument('--first', default='Smile', help='the name of the first character')
    parser.add_argument('--second', default='Strong', help='the name of the second character')
    parser.add_argument('--recoloring', default=False, action=argparse.BooleanOptionalAction,
        help='to recolor images')
    parser.add_argument('--swap_styles', default=False, action=argparse.BooleanOptionalAction,
        help='to swap image styles')
    parser.add_argument('--width', type=int, default=4, help='the width to align')
    args = parser.parse_args()

    imgs = read_images_and_labels('pixelated-characters.json', already_saved=True)

    img1 = imgs[f'{args.first}Face']
    img2 = imgs[f'{args.second}Face']

    if args.recoloring:
        img1 = ImageRecoloring(img1).apply(hue=get_random_beta(-1.0, 1.0))
        img2 = ImageRecoloring(img2).apply(hue=get_random_beta(-1.0, 1.0))

    img1.upscale(num=2)
    img2.upscale(num=2)

    if args.swap_styles:
        img_styled_1 = StyleTransfer(img1, img2, args.width).apply()
        img_styled_2 = StyleTransfer(img2, img1, args.width).apply()
        img1, img2 = img_styled_1, img_styled_2

    imageMorphing = ImageMorphing(img1, img2, args.width)

    for i in range(61):
        alpha = i / 60.0
        img = imageMorphing.get_morphed_image(alpha=alpha)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
