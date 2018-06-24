import os
import cv2

img_path = '/home/liuhy/Data/datasets/cow_face/testset/'
imgs = os.listdir(img_path)

for a_img in imgs:
    img = cv2.imread(os.path.join(img_path, a_img))
    img = cv2.resize(img, (182, 182))
    cv2.imwrite(os.path.join(img_path, a_img), img)
