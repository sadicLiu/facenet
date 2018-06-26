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
│   └── pairs.txt   // lfw数据集的pairs,这里面是选出来的一些正负样本的pair,主要是用在验证模型准确率
├── __init__.py
├── LICENSE.md
├── notes.md
├── README.md
├── requirements.txt
├── src
│   ├── align
│   ├── calculate_filtering_metrics.py
│   ├── classifier.py   // 用已经训练好的模型提取特征,训练自己的人脸分类器
│   ├── compare.py      // 比较多张给出的图片的欧氏距离
│   ├── decode_msceleb_dataset.py
│   ├── download_and_extract.py
│   ├── facenet.py      
│   ├── freeze_graph.py // 将训练好的模型中的变量变为常量,速度更快
│   ├── generative
│   ├── __init__.py
│   ├── lfw.py
│   ├── models      // cnn分类模型结构定义
│   ├── __pycache__
│   ├── train_softmax.py    // 使用softmax训练人脸识别模型
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
    python align_dataset_mtcnn.py \
    ~/Data/datasets/cow_face/raw/ \
    ~/Data/datasets/cow_face/align \
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
    3. Using a frozen graph significantly speeds up the loading of the model.

## Questions

- 01
    ```
    Q: 这个训练好的模型是如何应用到其他数据集的(看evaluation的代码)
    A: 直接用训练好的模型提取特征 -> 调用 `lfw.evaluate` -> 内部调用 `facenet.calculate_roc` 和 `facenet.calculate_val`
    ```
- 02
    ```
    Q: 代码中使用pretrained model的过程
    A: saver.restore(sess, pretrained_model_path)
    ```
- 03
    ```
    Q: compare的阈值是如何确定的,两个embeding距离是多少认为是同一个人?
    A: 可能的办法: 每头牛取三张图片(正脸+两个侧脸),提取特征,保存这三个特征向量,计算这三张图片的特征向量的欧氏距离,保存那个最大的距离值,用这种方法算出所有牛的最大距离值,求max(d1, d2, ..., dn),用这个值作为阈值.当有一张新来的图片,先算图片的特征向量,然后算这个特征向量和所有特征向量的距离,取距离最小值,如果这个最小值大于阈值,则新来的图片不在特征库中,若小于阈值,则新来的图片与距离最小的类属于同一类
    ```

## TODO

- [x] 在lfw数据集上验证人脸识别模型
- [x] 找一下牛脸数据和模型,同样用pairs模型验证一下
- [x] FFmpeg了解一下
- [x] 两个验证的数据集准确率差不多,但是validation rate差很多,为什么
- [x] 人脸的验证数据模型之前没见过,牛脸的验证数据模型之前都见过