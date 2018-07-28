#coding=utf-8
"""Export saved metegraph and checkpoints to tensorflow serving model."""
import argparse
import os
import shutil
import sys
import tensorflow as tf

import facenet

def main(args):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

        # Export facenet model
        saved_model_dir = os.path.join(args.saved_model_dir, str(args.version))
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
        os.mkdir(saved_model_dir)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            facenet.load_model(args.model_dir)

            print('Exporting trained model to ', saved_model_dir)
            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)

            # Get input and output tensors
            images_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

            sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: False})
            sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: False})

            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'images': tf.saved_model.utils.build_tensor_info(images_placeholder),
                    'phase': tf.saved_model.utils.build_tensor_info(phase_train_placeholder)
                },
                outputs={
                    'embeddings': tf.saved_model.utils.build_tensor_info(embeddings)
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'calculate_embeddings': prediction_signature
                }
            )

            builder.save()
            print('Successfully exported model to ', saved_model_dir)


def parse_arugments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (.ckpt) file.')
    parser.add_argument('version', type=int,
                        help='Version of current model.', default=1)
    parser.add_argument('saved_model_dir', type=str,
                        help='Directory for the exported graphdef protobuf (.pb).')

    return parser.parse_args(argv)
    pass


if __name__ == '__main__':
    main(parse_arugments(sys.argv[1:]))