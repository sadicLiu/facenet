# coding=utf-8
"""Perform classify task on given images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
import sys

import numpy as np
import tensorflow as tf
from scipy import misc

import facenet


def main(args):
    # args.image_files是图片的路径
    images = load_and_align_data(args.class_path, args.image_size)

    with tf.Graph().as_default():
         with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            logits = tf.get_default_graph().get_tensor_by_name("logits:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            logits_vec = sess.run(logits, feed_dict=feed_dict)
            result = np.argmax(logits_vec, axis=1) + 1
            print("Result: ", result)


def load_and_align_data(class_path, image_size):
    img_list = []

    imgs = os.listdir(class_path)
    for img in imgs:
        a_img = misc.imread(os.path.join(class_path, img), mode='RGB')
        aligned = misc.imresize(a_img, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)

    images = np.stack(img_list)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('class_path', type=str, help='Images to classify')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
