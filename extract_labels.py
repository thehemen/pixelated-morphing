from misc.image_labels import save_images
from pixel_art.data import read_images_and_labels

if __name__ == '__main__':
    """Read images with their region labels."""
    imgs = read_images_and_labels('pixelated-characters.json')
    save_images(imgs)
