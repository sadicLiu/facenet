#!/usr/bin/env bash

python ../src/train_softmax.py \
--logs_base_dir ~/logs/cowface/ \
--models_base_dir ~/models/cowface/ \
--data_dir ~/datasets/cowface/trainset/ \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 100 \
--batch_size 90 \
--keep_probability 0.4 \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file ../data/learning_rate_schedule_classifier_vggface2.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--validation_set_split_ratio 0.1 \
--validate_every_n_epochs 1 \
--pretrained_model /wls/majin/models/cowface/facenet/20180402-114759/model-20180402-114759.ckpt-275 \


