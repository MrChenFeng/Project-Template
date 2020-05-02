import cv2
import os
from tqdm import tqdm

video_path = '/home/chen/General_Datasets/KTH/videos/'
videos = os.listdir(video_path)
frame_path = '/home/chen/General_Datasets/KTH/frames/'
if not os.path.isdir(frame_path):
    os.mkdir(frame_path)
for i,video in enumerate(tqdm(videos)):
    name = video.split('.')[0]
    write_path = os.path.join(frame_path, name)
    if not os.path.isdir(write_path):
        os.mkdir(write_path)
    vid = cv2.VideoCapture(os.path.join(video_path,video))
    order = 0
    while vid.isOpened():
        order += 1
        ret, frame = vid.read()
        if not ret:
            break
        cv2.imwrite('{0}/image{1:04}.jpg'.format(write_path, order), frame)