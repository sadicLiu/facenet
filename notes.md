## Structure

```
.
├── contributed
│   ├── batch_represent.py
│   ├── clustering.py
│   ├── cluster.py
│   ├── export_embeddings.py
│   ├── face.py
│   ├── __init__.py
│   ├── predict.py
│   └── real_time_face_recognition.py
├── data
│   ├── images
│   ├── learning_rate_retrain_tripletloss.txt
│   ├── learning_rate_schedule_classifier_casia.txt
│   ├── learning_rate_schedule_classifier_msceleb.txt
│   ├── learning_rate_schedule_classifier_vggface2.txt
│   └── pairs.txt   // lfw数据集的pairs,在validate_on_lfw.py中用到了
├── __init__.py
├── LICENSE.md
├── notes.md
├── README.md
├── requirements.txt
├── src
│   ├── align
│   ├── calculate_filtering_metrics.py
│   ├── classifier.py
│   ├── compare.py
│   ├── decode_msceleb_dataset.py
│   ├── download_and_extract.py
│   ├── facenet.py
│   ├── freeze_graph.py
│   ├── generative
│   ├── __init__.py
│   ├── lfw.py
│   ├── models
│   ├── __pycache__
│   ├── train_softmax.py
│   ├── train_tripletloss.py
│   └── validate_on_lfw.py
├── s.txt
├── test
│   ├── batch_norm_test.py
│   ├── center_loss_test.py
│   ├── restore_test.py
│   ├── train_test.py
│   └── triplet_loss_test.py
├── tmp
│   ├── align_dataset.m
│   ├── align_dataset.py
│   ├── align_dlib.py
│   ├── cacd2000_split_identities.py
│   ├── dataset_read_speed.py
│   ├── deepdream.py
│   ├── detect_face_v1.m
│   ├── detect_face_v2.m
│   ├── download_vgg_face_dataset.py
│   ├── funnel_dataset.py
│   ├── __init__.py
│   ├── invariance_test.txt
│   ├── mnist_center_loss.py
│   ├── mnist_noise_labels.py
│   ├── mtcnn.py
│   ├── mtcnn_test_pnet_dbg.py
│   ├── mtcnn_test.py
│   ├── network.py
│   ├── nn2.py
│   ├── nn3.py
│   ├── nn4.py
│   ├── nn4_small2_v1.py
│   ├── pilatus800.jpg
│   ├── random_test.py
│   ├── rename_casia_directories.py
│   ├── seed_test.py
│   ├── select_triplets_test.py
│   ├── test1.py
│   ├── test_align.py
│   ├── test_invariance_on_lfw.py
│   ├── vggface16.py
│   ├── vggverydeep19.py
│   ├── visualize.py
│   ├── visualize_vggface.py
│   └── visualize_vgg_model.py
└── util
    └── plot_learning_curves.m
```


## Scripts For validation

- Face alignment

    ```
    python src/align/align_dataset_mtcnn.py \
    ~/Data/datasets/casia/CASIA-maxpy-clean/ \
    ~/Data/datasets/casia/casia_maxpy_mtcnnpy_182 \
    --image_size 182 \
    --margin 44
    ```
- Align the LFW dataset

    ```
    for N in {1..4}; do \
    python src/align/align_dataset_mtcnn.py \
    ~/Data/datasets/lfw/raw \
    ~/Data/datasets/lfw/lfw_mtcnnpy_160 \
    --image_size 160 \
    --margin 32 \
    --random_order \
    --gpu_memory_fraction 0.25 \
    & done
    ```
- Run the test

    ```
    python src/validate_on_lfw.py \
    ~/Data/datasets/lfw/lfw_mtcnnpy_160 \
    ~/Data/models/facenet/20180402-114759 \
    --distance_metric 1 \
    --use_flipped_images \
    --subtract_mean \
    --use_fixed_image_standardization
    ```

## Classifier training of inception resnet v1

- Face alignment

    ```
    python src/align/align_dataset_mtcnn.py \
    ~/datasets/casia/CASIA-maxpy-clean/ \
    ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
    --image_size 182 \
    --margin 44
    ```
- GPU version

    ```
    for N in {1..4}; do \
    python src/align/align_dataset_mtcnn.py \
    ~/datasets/casia/CASIA-maxpy-clean/ \
    ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
    --image_size 182 \
    --margin 44 \
    --random_order \
    --gpu_memory_fraction 0.25 \
    & done
    ```
- Start training

    ```
    python src/train_softmax.py \
    --logs_base_dir ~/Data/logs/facenet/ \
    --models_base_dir ~/Data/models/facenet/ \
    --data_dir ~/Data/datasets/casia/casia_maxpy_mtcnnalign_182_160/ \
    --image_size 160 \
    --model_def models.inception_resnet_v1 \
    --lfw_dir ~/Data/datasets/lfw/lfw_mtcnnalign_160/ \
    --optimizer ADAM \
    --learning_rate -1 \
    --max_nrof_epochs 150 \
    --keep_probability 0.8 \
    --random_crop \
    --random_flip \
    --use_fixed_image_standardization \
    --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
    --weight_decay 5e-4 \
    --embedding_size 512 \
    --lfw_distance_metric 1 \
    --lfw_use_flipped_images \
    --lfw_subtract_mean \
    --validation_set_split_ratio 0.05 \
    --validate_every_n_epochs 5 \
    --prelogits_norm_loss_factor 5e-4
    ```

    1. If the parameter lfw_dir is set to point to a the base directory of the LFW dataset the model is evaluated on LFW once every 1000 batches. If no evaluation on LFW is desired during training it is fine to leave the lfw_dir parameter empty.
    2. The training will continue until the max_nrof_epochs is reached or training is terminated from the learning rate schedule file.

## Train a classifier on LFW

- Start training

    ```
    python src/classifier.py TRAIN /home/david/datasets/lfw/lfw_mtcnnalign_160 /home/david/models/model-20170216-091149.pb ~/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset
    ```
    - mode=TRAIN
    - data_dir=/home/david/datasets/lfw/lfw_mtcnnalign_160
    - model=/home/david/models/model-20170216-091149.pb
    - classifier_filename=~/models/lfw_classifier.pkl
    - --batch_size 1000 
    - --min_nrof_images_per_class 40
    - --nrof_train_images_per_class 35
    - --use_split_dataset

- Start testing

    ```
    python src/classifier.py CLASSIFY ~/datasets/lfw/lfw_mtcnnalign_160 ~/models/model-20170216-091149.pb ~/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset
    ```

## Train a classifier on your own dataset

- Start training

    ```
    python src/classifier.py TRAIN ~/datasets/my_dataset/train/ ~/models/model-20170216-091149.pb ~/models/my_classifier.pkl --batch_size 1000
    ```

- Start testing

    ```
    python src/classifier.py CLASSIFY ~/datasets/my_dataset/test/ ~/models/model-20170216-091149.pb ~/models/my_classifier.pkl --batch_size 1000
    ```

- Notes
    This code is aimed to give some inspiration and ideas for how to use the face recognizer, but it is by no means a useful application by itself. Some additional things that could be needed for a real life application include:

    1. Include face detection in a face detection and classification pipe line
    2. Use a threshold for the classification probability to find unknown people instead of just using the class with the highest probability

## Questions

- 01
    ```
    Q: 这个训练好的模型是如何应用到其他数据集的(看evaluation的代码)
    A: 直接用训练好的模型提取特征 -> 调用 `lfw.evaluate` -> 内部调用 `facenet.calculate_roc` 和 `facenet.calculate_val`
    ```


2. 代码中使用pretrained model的过程
3. 1:1 1:N 的过程