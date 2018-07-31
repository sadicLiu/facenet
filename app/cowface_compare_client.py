# coding=utf-8
"""Tensorflow Serving Client used for serving compare request.
"""
import argparse
import copy
import os
import pickle
import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from grpc.beta import implementations
from scipy import misc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import facenet


def main(args):
    # Init tensorflow serving stuff
    host, port = args.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    THRESHOLD_SAME = 0.1

    # Load embeddings
    print('Loading embeddings')
    with open(args.emb_file, 'rb') as infile:
        emb_array, labels, paths = pickle.load(infile)

    # Load and detect cowface image
    cowface_detect = detect_cowface(args.image, args.image_size, stub, args.detect_model, args.label_map_path,
                                    args.num_classes)

    # Extract feature of given image to compare.
    embedding = extract_feature(cowface_detect, stub, args.verification_model)

    # TODO: Compare the extracted feature with all saved features.
    dis, lable, path = compare(embedding, args.emb_file)
    if dis < THRESHOLD_SAME:
        print('Found the cow. Nearest distance is: {}. Nearest image: [{}]'.format(dis, path))
    else:
        print('Not found the cow. Nearest distance is: {}. Nearest image: [{}]'.format(dis, path))


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def detect_cowface(image_path, image_size, stub, detect_model):
    """Load cowface image (only support one image) and detect cowface using detection model"""
    image = Image.open(os.path.expanduser(image_path))
    # The array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Detect cowface
    request = predict_pb2.PredictRequest()
    request.model_spec.name = detect_model
    request.model_spec.signature_name = 'serving_default'
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_np_expanded, shape=image_np_expanded.shape))
    result = stub.Predict(request, 5.0)  # 5 seconds timeout

    boxes = np.array(result.outputs['detection_boxes'].folat_val).reshape(
        result.outputs['detection_boxes'].tensor_shape.dim[0].size,
        result.outputs['detection_boxes'].tensor_shape.dim[1].size,
        result.outputs['detection_boxes'].tensor_shape.dim[2].size,
    )[0]
    classes = np.array(result.outputs['detection_classes'].float_val).reshape((1, -1))[0]
    scores = np.array(result.outputs['detection_scores'].float_val).reshape((1, -1))[0]

    box = boxes[0]
    print("detect box: ", box)
    img_width, img_height = image.size
    ymin = box[0] * img_height
    ymax = box[2] * img_height
    xmin = box[1] * img_width
    xmax = box[3] * img_width

    crop = np.array(image)[max(0, int(ymin)): int(ymax), max(0, int(xmin)): int(xmax)]

    aligned = misc.imresize(crop, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)

    return prewhitened


def extract_feature(image, stub, verification_model):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = verification_model
    request.model_spec.signature_name = 'calculate_embeddings'

    image_expanded = np.expand_dims(image, axis=0)

    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(image_expanded, shape=image_expanded.shape))
    request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
    result = stub.Predict(request, 5.0)    # 5.0 seconds
    embedding = result.outputs['embeddings'].float_val

    return np.asarray(embedding)


def  compare(embedding, emb_file):
    # Load embeddings
    print('Loading embeddings')
    with open(emb_file, 'rb') as infile:
        emb_array, labels, paths = pickle.load(infile)

    # Calculate min distance of the embedding in emb_array
    diff = np.subtract(emb_array, embedding)
    distances = np.sum(np.square(diff), 1)
    min_idx = np.argmin(distances)

    min_dis = distances[min_idx]
    min_lab = labels[min_idx]
    min_path = paths[min_idx]

    return min_dis, min_lab, min_path

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
    parser.add_argument('detect_model', type=str,
                        help='Detection model in tf serving', default='cowface_detect')
    parser.add_argument('verification_model', type=str,
                        help='Verification model in tf serving', default='cowface_verification')
    parser.add_argument('--dataset_dir', type=str,
                        help='If using the EXTRACT mode, then dataset_dir should be provided to save features.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--embedding_size', type=int,
                        help='The length of embedding vector.', default=128)
    parser.add_argument('--emb_file', type=str,
                        help='Path to saved embedding file.', default='./embeddings.pkl')
    parser.add_argument('--label_map_path', type=str,
                        help='Path to label map file.', default='./cowface_label_map.pbtxt')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
