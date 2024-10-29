import cv2
import numpy as np
from pixel_art.morphing import ImageMorphing
from pixel_art.recoloring import ImageRecoloring
from pixel_art.style_transfer import StyleTransfer
from misc.misc import get_random_beta, pixelate, apply_over_channels

class ImGeneration:
    """
        ImGeneration creates a list of images
        by using the procedural generation.
    """
    def __init__(self, imgs, categoryList, paramDict):
        self.imgs = imgs
        self.categoryList = categoryList

        self.recolor_value = paramDict['recolor_value']
        self.change_style_value = paramDict['change_style_value']
        self.alpha = paramDict['alpha']
        self.beta = paramDict['beta']
        self.width = paramDict['width']
        self.upscale_num = paramDict['upscale_num']
        self.pixelate = paramDict['pixelate']
        self.align_width = self.width * 2 ** self.upscale_num

    def get_character_name(self, characters_used, only_character_chosen=False):
        """Get a random character name."""
        character_name = self.categoryList.get_character(characters_used, only_character_chosen)
        characters_used.append(character_name)
        return character_name

    def generate_image(self, filename=None, color_space='bgr'):
        """Generate images by the morphing algorithm."""
        characters_used = []

        img1 = self.imgs[self.get_character_name(characters_used)]
        img2 = self.imgs[self.get_character_name(characters_used)]
        img12 = [img1, img2]

        for j, img in enumerate(img12):
            if np.random.random() < self.recolor_value:
                img12[j] = ImageRecoloring(img).apply(hue=get_random_beta(-1.0, 1.0))

        for j, img in enumerate(img12):
            img.upscale(num=self.upscale_num)

            if np.random.random() < self.change_style_value:
                img_style_from = self.imgs[self.get_character_name(characters_used, only_character_chosen=True)]
                img12[j] = StyleTransfer(img, img_style_from, self.width).apply()

        img1, img2 = img12
        imageMorphing = ImageMorphing(img1, img2, self.align_width)

        alpha = get_random_beta(0.0, 1.0, self.alpha, self.beta)
        img = imageMorphing.get_morphed_image(alpha=alpha)

        if self.pixelate:
            shape = np.array(img.shape)
            shape = (shape / self.width).round().astype(int)
            img = apply_over_channels(pixelate, img, shape)

        if color_space == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        if filename:
            cv2.imwrite(filename, img)
        else:
            return img
