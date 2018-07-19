#coding=utf-8
"""
Picking images.
"""
import os
import numpy as np

PATH = '/home/liuhy/Downloads/test'
PICK_NUM = 40


def down_sample(smp_rate):
    """图片很多，每隔几张降采样一次"""
    imgs = np.sort(os.listdir(PATH))
    num_imgs = len(imgs)
    for i in range(num_imgs):
        if i % smp_rate != 0:
            os.remove(os.path.join(PATH, imgs[i]))

    picked_imgs = os.listdir(PATH)
    for i in range(len(picked_imgs)):
        img_name = str(i+1) + '.jpg'
        os.rename(os.path.join(PATH, picked_imgs[i]), os.path.join(PATH, img_name))

def rand_pick():
    """图片较少，随机选40张"""
    imgs = np.sort(os.listdir(PATH))
    num_imgs = len(imgs)
    indices = range(num_imgs)
    np.random.shuffle(indices)

    for i in range(num_imgs - PICK_NUM):
        img_path = os.path.join(PATH, imgs[indices[i]])
        os.remove(img_path)


def main():
    """Picking start from here"""
    num_imgs = len(os.listdir(PATH))
    smp_rate = (num_imgs / PICK_NUM) if (num_imgs / PICK_NUM > 1) else 0

    if smp_rate > 1:
        down_sample(smp_rate)
    else:
        rand_pick()


if __name__ == '__main__':
    main()
