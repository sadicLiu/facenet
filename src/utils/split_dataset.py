"""
Split trainset and testset
"""
import os
import shutil
import numpy as np

SOURCE = '/home/liuhy/Downloads/dataset/src'
TRAINSET = '/home/liuhy/Downloads/dataset/trainset'
TESTSET = '/home/liuhy/Downloads/dataset/testset'
NUM_TRAIN = 140.0
NUM_TEST = 40.0


def main():
    assert os.path.exists(SOURCE), "Source path doesn't exists!"
    print("Start processing...")

    if os.path.exists(TRAINSET):
        shutil.rmtree(TRAINSET)
    os.mkdir(TRAINSET)
    if os.path.exists(TESTSET):
        shutil.rmtree(TESTSET)
    os.mkdir(TESTSET)

    classes = os.listdir(SOURCE)
    num_classes = len(classes)
    num_train = int(num_classes * (NUM_TRAIN / (NUM_TRAIN + NUM_TEST)))

    indices = range(1, num_classes + 1)
    np.random.shuffle(indices)

    for i in range(0, num_train):
        a_class_src = os.path.join(SOURCE, str(indices[i]))
        a_class_tgt = os.path.join(TRAINSET, str(i + 1))
        shutil.copytree(a_class_src, a_class_tgt)
    for j in range(num_train, num_classes):
        a_class_src = os.path.join(SOURCE, str(indices[j]))
        a_class_tgt = os.path.join(TESTSET, str(j + 1 - num_train))
        shutil.copytree(a_class_src, a_class_tgt)

    print("Success.")

if __name__ == '__main__':
    main()