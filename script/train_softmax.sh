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
--validation_set_split_ratio 0.01 \
--validate_every_n_epochs 5

# --lfw_dir ~/datasets/lfw/lfw_mtcnnalign_160/ \
# --lfw_distance_metric 1 \
# --lfw_use_flipped_images \
# --lfw_subtract_mean \
