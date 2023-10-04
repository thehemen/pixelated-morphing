import cv2
import argparse
from pixel_art.data import read_images_and_labels

if __name__ == '__main__':
    """Show a pixelated character."""
    parser = argparse.ArgumentParser(description='Show a pixelated character')
    parser.add_argument('--name', default='Smile', help='the name of character')
    args = parser.parse_args()

    imgs = read_images_and_labels('pixelated-characters.json', already_saved=True)
    img = imgs[f'{args.name}Face']

    while True:
        cv2.imshow(args.name, img.get_image())
        k = cv2.waitKey(0)

        if k == 82:
            img.upscale()
        elif k == 84:
            img.downscale()
        elif k == 27:
            break

    cv2.destroyAllWindows()
