import os
import shutil


base_path = "D:\\Users\\LIUHONGYU393\\Desktop\\cow_face"
raw_video_path = os.path.join(base_path, 'cows_video2')

videos = os.listdir(raw_video_path)
video_num = len(videos)

print(video_num)

start_num = 120  # 之前已经有了120头牛的数据，从120开始
for i in range(video_num):
    video_name = videos[i]
    new_path = os.path.join(raw_video_path, str(i + 120))
    os.mkdir(new_path)
    shutil.move(os.path.join(raw_video_path, video_name), os.path.join(new_path, '1.mp4'))



