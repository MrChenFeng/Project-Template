import os
import cv2
from tqdm import tqdm

def video_to_frames(video_path, frame_path):
    """
    For each videos, extract frames and save to frame dirs
    """
    if not os.path.isdir(frame_path):
        os.mkdir(frame_path)
    files = os.listdir(video_path)
    for i in tqdm(files):
        a_path = os.path.join(video_path, i)
        videos = sorted(os.listdir(a_path))
        if not os.path.isdir(os.path.join(frame_path, i)):
            os.mkdir(os.path.join(frame_path, i))
        for j in range(len(videos)):
            now_path = os.path.join(frame_path, i,'video{0:03}'.format(j+1))
            if not os.path.isdir(now_path):
                os.mkdir(now_path)
            vid = cv2.VideoCapture(os.path.join(a_path,videos[j]))
            order = 0
            while vid.isOpened():
                order += 1
                ret, frame = vid.read()
                if not ret:
                    break
                cv2.imwrite('{0}/image{1:03}.jpg'.format(now_path, order), frame)
    return None

if __name__ == '__main__':
    # Calling the function
    video_to_frames("/home/chen/Video_Disentanled_Representation/Datasets/HMDB51","/home/chen/Video_Disentanled_Representation/Datasets/HMDB51Frames")