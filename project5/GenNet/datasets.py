from __future__ import division

import math
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import scipy.misc

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class DataSet(object):
    def __init__(self, data_path, image_size=128):
        self.root_dir = data_path
        self.imgList = [f for f in os.listdir(data_path) if any(f.endswith(ext) for ext in IMG_EXTENSIONS)]
        self.imgList.sort()
        self.image_size = image_size
        self.images = np.zeros((len(self.imgList), image_size, image_size, 3)).astype(float)
        print('Loading dataset: {}'.format(data_path))
        for i in range(len(self.imgList)):
            image = Image.open(os.path.join(self.root_dir, self.imgList[i])).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image).astype(float)
            self.images[i] = image
        print('Data loaded, shape: {}'.format(self.images.shape))

    def data(self):
        return self.images

    def mean(self):
        return np.mean(self.images, axis=(0, 1, 2, 3))

    def to_range(self, low_bound, up_bound):
        min_val = self.images.min()
        max_val = self.images.max()
        return low_bound + (self.images - min_val) / (max_val - min_val) * (up_bound - low_bound)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.imgList)


def merge_images(images, space=0, mean_img=None):
    num_images = images.shape[0]
    canvas_size = int(np.ceil(np.sqrt(num_images)))
    h = images.shape[1]
    w = images.shape[2]
    canvas = np.zeros((canvas_size * h + (canvas_size-1) * space,  canvas_size * w + (canvas_size-1) * space, 3), np.uint8)

    for idx in range(num_images):
        image = images[idx]
        if mean_img:
            image += mean_img
        i = idx % canvas_size
        j = idx // canvas_size
        min_val = np.min(image)
        max_val = np.max(image)
        image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        canvas[j*(h+space):j*(h+space)+h, i*(w+space):i*(w+space)+w,:] = image
    return canvas


def save_images(images, file_name, space=0, mean_img=None):
    scipy.misc.imsave(file_name, merge_images(images, space, mean_img))