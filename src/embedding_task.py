# coding=utf-8
"""Extract all embeddings in given dataset(EXTRACT mode) or perform classification task using embeddings.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import math
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from scipy import misc

import facenet


def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            emb_file = 'embeddings.pkl'

            # Extract embeddings in trainset
            if (args.mode == 'EXTRACT'):
                np.random.seed(seed=666)
                dataset = facenet.get_dataset(args.data_dir)

                # Check that there are at least one training image per class
                for cls in dataset:
                    assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

                paths, labels = facenet.get_image_paths_and_labels(dataset)

                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))

                # Load the model
                print('Loading feature extraction model')
                facenet.load_model(args.model)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * args.batch_size
                    end_index = min((i + 1) * args.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, args.image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                with open(emb_file, 'wb') as outfile:
                    pickle.dump((emb_array, labels, paths), outfile)
                print('Embeddings save to file "%s"' % emb_file)
            # Perform compare task
            else:
                # Load embeddings
                print('Loading embeddings')
                with open(emb_file, 'rb') as infile:
                    emb_array, labels, paths = pickle.load(infile)

                # Load the model
                print('Loading feature extraction model')
                facenet.load_model(args.model)
                images = load_and_align_data(args.image_files, args.image_size)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)

                nrof_images = len(args.image_files)

                print('Images and embeddings:')
                for i in range(nrof_images):
                    min_result = min_distance(emb[i], emb_array)
                    min_result_label = labels[min_result]
                    min_result_path = paths[min_result]
                    print('Compared image: %s, compared result: %s, label: %s, path: %s'
                          % (args.image_files[i], min_result, min_result_label, min_result_path))


def min_distance(embedding, emb_array):
    """Calculate min distance of the embedding in emb_array"""
    diff = np.subtract(emb_array, embedding)
    distance = np.sum(np.square(diff), 1)
    min = np.argmin(distance)
    return min


def load_and_align_data(image_paths, image_size):
    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')

        # TODO: 这里之后可能会把detection的过程加进来
        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['EXTRACT', 'COMPARE'],
                        help='Indicates if trainset embeddings should be extract and save to disk' +
                             'or performing compare on given images.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--data_dir', type=str,
                        help='If using the EXTRACT mode, then data_dir should supplied.')
    parser.add_argument('--image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=16)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
