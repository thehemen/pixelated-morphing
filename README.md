# Pixelated Morphing

This code allows you to create different pixelated character images by using the morphing algorithm.
A visual description of the process is described on [Medium](https://medium.com/@thehemen/generative-art-of-pixelated-characters-tips-and-practices-69f38b63e49c).

The generation by the DCGAN model is available as well.
A description of the process is presented on [Medium](https://medium.com/@thehemen/generative-art-of-pixelated-characters-usage-of-dcgan-model-f6682a6decff).

## How to Generate Images

As a baseline, a set of 100 images similar to [Minecraft's ones](https://minecraft.fandom.com/wiki/Minecraft_Wiki) are used.

To run this algorithm, you need to install requirements:

```sh
pip install -r requirements.txt
```

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
python3 show_img.py --name [character name]
```

To upscale/downscale an image, use Up/Down key arrows.

Finally, you can generate new images (that will be saved in characters.zip):

```sh
python3 generate_images.py --n [image_number] \
--recolor_value [recoloring probability] \
--change_style_value [style changing probability] \
--alpha [alpha value of beta distribution] \
--beta [beta value of beta distribution] \
--width [align width] \
--upscale_num [image zoom number] \
--n_jobs [number of parallel processes] \
--pixelate [use pixelation]
```

So, the work is done.

## How to Visualize Images

There are few additional scripts that allow you to check out the different aspects of the algorithm.

For instance, if you've generated images with the `--pixelate` option turned on, you can upscale them by using this command:

```sh
python3 unpixelate_images.py \
--input [initial image folder] \
--output [unpixelated image folder] \
--width [align width]
```

To visualize the image recoloring by its [HCL](https://hclwizard.org/) color channels, you can use this [DearPyGUI](https://github.com/hoffstadt/DearPyGui) based application:

```sh
python3 show_recoloring.py --name [character name]
```

To see, how an image's style transfer is applied, run this one:

```sh
python3 show_style_transfer.py \
--first [character name for which the style is transferred] \
--second [character name from which the style is transferred] \
--width [align width]
```

Finally, you can run this script to visualize the morphing process with recoloring and swap style effects:

```sh
python3 show_morphing.py \
--first [the first character name] \
--second [the second character name] \
--recoloring [use recoloring] \
--swap_styles [use swap styles] \
--frame_num [number of frames] \
--delay [delay between frames in ms] \
--width [align width]
```

## How to Train DCGAN Model

In addition to the morphing algorithm, the neural network's generation is added.

To train a DCGAN model, run this command:

```sh
python3 train_dcgan.py \
--n [number of images] \
--seed [seed value] \
--batch_size [size of the batch] \
--num_epochs [number of epochs] \
--checkpoint_epoch [number of epochs when the model is saved] \
--img_epoch [number of epochs when the image is saved] \
--change_image_num [number of image history epochs] \
--change_image_value [probability of image change] \
--noise_value [max value of noise added to labels] \
--workers [number of parallel processes] \
--gpu [use the GPU device]
```

## How to Test DCGAN model

To test a DCGAN model, run this command:

```sh
python3 test_dcgan.py \
--model_name [filename of the model] \
--seed_first [first image seed value] \
--seed_second [second image seed value] \
--width [width to align] \
--frame_num [number of frames] \
--delay [delay between frames in ms] \
--gpu [use the GPU device] \
--image [use the single image]
```

Good luck!
