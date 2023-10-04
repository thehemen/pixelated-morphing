import cv2
import argparse
import dearpygui.dearpygui as dpg
from pixel_art.recoloring import ImageRecoloring
from pixel_art.data import read_images_and_labels

def cv2_to_mv(img):
    """Convert an image from cv2 to mv format."""
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img.flatten() / 255.0

if __name__ == '__main__':
    """Show an image recoloring by LCh color components."""
    dpg.create_context()
    dpg.create_viewport()
    dpg.setup_dearpygui()

    parser = argparse.ArgumentParser(description='Show recolored character image.')
    parser.add_argument('--name', default='Smile', help='the name of character')
    args = parser.parse_args()

    imgs = read_images_and_labels('pixelated-characters.json', already_saved=True)
    img = imgs[f'{args.name}Face']
    img.upscale(num=2)

    img_initial = img.get_image()
    height, width = img_initial.shape[:2]

    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(width=width, height=height, default_value=cv2_to_mv(img_initial), tag='img')

    def image_callback(sender):
        img_recoloring = ImageRecoloring(img)

        img_new = img_recoloring.apply(
            luminance=dpg.get_value('luminance'),
            chroma=dpg.get_value('chroma'),
            hue=dpg.get_value('hue')).get_image()

        dpg.set_value('img', cv2_to_mv(img_new))

    with dpg.window(label=args.name):
        dpg.add_slider_float(label='Luminance', tag='luminance', default_value=0.0,
            min_value=-1.0, max_value=1.0, callback=image_callback)
        dpg.add_slider_float(label='Chroma', tag='chroma', default_value=0.0,
            min_value=-1.0, max_value=1.0, callback=image_callback)
        dpg.add_slider_float(label='Hue', tag='hue', default_value=0.0,
            min_value=-1.0, max_value=1.0, callback=image_callback)

        with dpg.drawlist(width=width, height=height):
            dpg.draw_image('img', (0, 0), (width, height))

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
