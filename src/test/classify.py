import copy
import os

import numpy as np
from scipy import misc

import facenet

PATH = '/home/liuhy/Pictures'
SIZE = 160


def load_and_align_data(image_paths, image_size):
    # tmp_image_paths = copy.copy(image_paths)
    tmp_image_paths = os.listdir(PATH)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.join(PATH, image), mode='RGB')
        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def main():
    images = load_and_align_data(PATH, SIZE)
    print(images.shape)     # (8, 160, 160, 3)


if __name__ == '__main__':
    main()
