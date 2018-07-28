# coding=utf-8
"""Tensorflow Serving Client used for serving compare request.
"""
import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import  prediction_service_pb2


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def main(args):
    pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('server', type=str,
                        help='PredictionService host:port.', default='0.0.0.0:9001')
    parser.add_argument('image', type=str,
                        help='JPG image path to compare.')
    parser.add_argument('save_path', type=str,
                        help='Save path for output image.', default='./')
    parser.add_argument('num_classes', type=int,
                        help='Classes number in detection model.', default=1)
    parser.add_argument('mode', type=str, choices=['EXTRACT', 'COMPARE'],
                        help='Indicates if trainset embeddings should be extract and save to disk' +
                             'or performing compare on given images.')
    parser.add_argument('--data_dir', type=str,
                        help='If using the EXTRACT mode, then data_dir should be provided to save features.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--detect_model', type=str,
                        help='Detection model in tf serving', default='cowface_detect')
    parser.add_argument('verification_model', type=str,
                        help='Verification model ')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
