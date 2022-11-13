from __future__ import print_function
import numpy as np
import skimage.transform as trans
import sys

np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)

# ########################configuration########################
OBJECT = [120, 0, 0]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([OBJECT, Unlabelled])
CLASS_NAME = ['OBJECT', 'None']  # You must define by yourself

COLOR_USE = 'grayscale'

IMAGE_SIZE_TEST = 256 * 256

IMG_SIZE_INPUT = (256, 256)


###############################################################


def adjust_data(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255.
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        mask[(mask != 0.) & (mask != 255.) & (mask != 128.)] = 0.
        new_mask = np.zeros(mask.shape + (num_class,))
        new_mask[mask == 255., 0] = 1
        new_mask[mask == 128., 1] = 1
        new_mask[mask == 0., 2] = 1
        mask = new_mask

    elif (np.max(img) > 1):
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def test_generator(images, num_image=1, target_size=IMG_SIZE_INPUT, flag_multi_class=True, as_gray=True):
    for i in range(len(images)):
        # img = img / 255.
        img = trans.resize(images[i], target_size)
        img = np.reshape(img, img.shape + (1,)) if (flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


# draw imgs in labelVisualize and save results in saveResult
def label_visualize(num_class, color_dict, img):
    img_out = np.zeros(img[:, :, 0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i, j])
            img_out[i, j] = color_dict[index_of_class]
    return img_out


def save_result(npyfile, flag_multi_class=True, num_class=2):
    count = 1
    images = []
    for i, item in enumerate(npyfile):
        if flag_multi_class:
            img = label_visualize(num_class, COLOR_DICT, item)
            img = img.astype(np.uint8)
            images.append(img)
        else:
            img = item[:, :, 0]
            # print(np.max(img),np.min(img))
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            # print(np.max(img),np.min(img))
            img = img * 255.
            images.append(img)
        count += 1
    return images
