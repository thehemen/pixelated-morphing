from copy import deepcopy
from misc.colors import adjust_colors

classes = ['Head', 'Eye', 'Nose', 'Mouth', 'Hair', 'Helmet', 'Ear', 'Eyebrow']

class ImageRecoloring:
	"""
		ImageRecoloring is used to recolor an image by its LCh components (Luminance, Chroma, Hue).
	"""

	def __init__(self, img):
		"""Save an image as a class field."""
		self.img = img

	def apply(self, luminance=0.0, chroma=0.0, hue=0.0):
		"""Apply recoloring to the image by its LCh color components.

		Keyword arguments:
		luminance -- the luminance component (default 0.0)
		chroma -- the chroma component (default 0.0)
		hue -- the hue component (default 0.0)
		"""
		img_recolored = deepcopy(self.img)

		for class_name in classes:
			if class_name not in self.img.get_classes():
				continue

			im_regions = self.img[class_name].get_regions()
			imgs = [img.image_area for img in im_regions]

			is_pair_class = len(im_regions) == 2

			for i, img in enumerate(imgs):
				img_new = adjust_colors(img, luminance, chroma, hue)

				if not is_pair_class:
					img_recolored[class_name].image_area = img_new
				else:
					img_recolored[class_name][i].image_area = img_new

		return img_recolored
