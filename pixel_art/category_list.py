import json
import itertools
import numpy as np

class CategoryList:
    """
        CategoryList implements a category list that makes it easier
        to choose from assets groupped into categories a random one.
    """
    def __init__(self, filename, only_rectangular_used):
        """CategoryList constructor.

        Fields:
        assets_by_category -- a dict of assets accessible by categories
        category_by_asset -- a dict of categories accessible by assets
        """
        with open(filename, 'r') as f:
            json_dict = json.load(f)
            self.assets_by_category = {}

            for k, v in json_dict.items():
                if only_rectangular_used and not v["rect"]:
                    continue

                self.assets_by_category[k] = v["images"]

        self.category_by_asset = {}

        for category, assets in self.assets_by_category.items():
            for asset in assets:
                self.category_by_asset[asset] = category

    def get_character(self, characters_used=[], only_character_used=False):
        """Get a character (an asset) from those not yet used.

        Keyword arguments:
        characters_used -- a list of characters that are already used
        only_character_used -- a flag to choose one character from all ones or not
        """
        # Initialize a dict of assets by categories from its class field.
        assets_by_category = {k: v[:] for k, v in self.assets_by_category.items()}

        # Remove from a dict the characters already used.
        for asset in characters_used:
            category = self.category_by_asset[asset]
            assets_by_category[category].remove(asset)

            # If a category is no longer used, remove it as well.
            if len(assets_by_category[category]) == 0:
                assets_by_category.pop(category)

        if not only_character_used:
            # Randomly choose a category, then choose a character.
            category = np.random.choice(list(assets_by_category.keys()))
            character = np.random.choice(assets_by_category[category])
        else:
            # Get a list of all assets, then choose a random character.
            assets = list(itertools.chain.from_iterable(assets_by_category.values()))
            character = np.random.choice(assets)

        # Return a random character.
        return character
