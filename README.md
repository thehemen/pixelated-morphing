# Pixelated Morphing

This code allows you to create different pixelated character images by using the morphing algorithm.
A visual description of the process is presented on [Medium](https://medium.com/@thehemen/generative-art-of-pixelated-characters-tips-and-practices-69f38b63e49c).

## How to Generate Images

As a baseline, a set of 16 images similar to [Minecraft's ones](https://minecraft.fandom.com/wiki/Minecraft_Wiki) are used.

First, you need to get aligned versions of initial images:

```sh
python3 add_aligned_imgs.py
```

Second, you need to label these images by [Computer Vision Annotation Tool](https://github.com/opencv/cvat), then save them in [COCO](https://cocodataset.org/) format.

After that, you should extract the annotations of the resulting images:

```sh
python3 extract_labels.py
```

Also, the categories of these images should be mentioned in categories.json.

You may see some missing areas in images. You can fill them with the help of any graphic editor (for example, [GIMP](https://www.gimp.org/)).

Then, you can visualize an image:

```sh
python3 show_img.py --name [character_name]
```

To upscale/downscale an image, use Up/Down key arrows.

Finally, you can generate new images (that will be saved in characters.zip):

```sh
python3 generate_images.py --n [image_number] \
--recolor_value [recoloring_probability] \
--change_style_value [style_changing_probability] \
--alpha [alpha_value_of_beta_distribution] \
--beta [beta_value_of_beta_distribution] \
--width [align_width] \
--upscale_num [image_zoom_number] \
--n_jobs [number_of_parallel_processes] \
--pixelate [use_pixelation]
```

So, the work is done.

## How to Visualize Images

There are few additional scripts that allow you to check out the different aspects of the algorithm.

For instance, if you've generated images with the `--pixelate` option turned on, you can upscale them by using this command:

```sh
python3 unpixelate_images.py \
--input [initial_image_folder] \
--output [unpixelated_image_folder] \
--width [align_width]
```

To visualize the image recoloring by its [HCL](https://hclwizard.org/) color channels, you can use this [DearPyGUI](https://github.com/hoffstadt/DearPyGui) based application:

```sh
python3 show_recoloring.py --name [character_name]
```

To see, how an image's style transfer is applied, run this:

```sh
python3 show_style_transfer.py \
--first [character_name_for_which_the_style_is_transferred] \
--second [character_name_from_which_the_style_is_transferred] \
--width [align_width]
```

Finally, you can run this script to visualize the morphing process with recoloring and swap style effects:

```sh
python3 show_morphing.py \
--first [the_first_character_name] \
--second [the_second_character_name] \
--recoloring [use_recoloring] \
--swap_styles [use_swap_styles] \
--width [align_width]
```

Good luck!
