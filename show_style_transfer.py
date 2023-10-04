import cv2
import argparse
from pixel_art.data import read_images_and_labels
from pixel_art.style_transfer import StyleTransfer

if __name__ == '__main__':
    """Show a style transfer between characters."""
    parser = argparse.ArgumentParser(description='Show style transfer between characters.')
    parser.add_argument('--first', default='Smile', help='the character\'s name for which the style is transferred')
    parser.add_argument('--second', default='Strong', help='the character\'s name from which the style is transferred')
    parser.add_argument('--width', type=int, default=4, help='the width to align')
    args = parser.parse_args()

    imgs = read_images_and_labels('pixelated-characters.json', already_saved=True)

    img1 = imgs[f'{args.first}Face']
    img2 = imgs[f'{args.second}Face']

    img1.upscale(num=2)
    img2.upscale(num=2)

    styleTransfer = StyleTransfer(img1, img2, args.width)
    img_styled = styleTransfer.apply()

    cv2.imshow('Image', img_styled.get_image())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
