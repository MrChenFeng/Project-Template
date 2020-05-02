import cv2
from scipy.io import loadmat
import numpy as np
import os
from tqdm import tqdm

vid_path = '/home/chen/Video_Disentanled_Representation/Datasets/Penn_Action/frames'
ann_path = '/home/chen/Video_Disentanled_Representation/Datasets/Penn_Action/labels'
pad = 5

videos = sorted(os.listdir(vid_path))

for video in tqdm(videos):
    ff = loadmat('{}/{}.mat'.format(ann_path, video))
    bboxes = ff['bbox']
    posey = ff['y'].astype(float)
    posex = ff['x'].astype(float)
    visib = ff['visibility']
    imgs = sorted([f for f in os.listdir('{}/{}'.format(vid_path, video)) if f.endswith('.jpg')])
    box = np.zeros((4,), dtype='int32')
    bboxes = bboxes.round().astype('int32')

    if len(imgs) > bboxes.shape[0]:
        bboxes = np.concatenate((bboxes, bboxes[-1][None]), axis=0)

    box[0] = bboxes[:, 0].min()
    box[1] = bboxes[:, 1].min()
    box[2] = bboxes[:, 2].max()
    box[3] = bboxes[:, 3].max()



    for j in range(len(imgs)):
        img = cv2.imread('{}/{}/{}'.format(vid_path, video, imgs[j]))

        y1 = box[1] - pad
        y2 = box[3] + pad
        x1 = box[0] - pad
        x2 = box[2] + pad

        h = y2 - y1 + 1
        w = x2 - x1 + 1

        if h > w:
            left_pad = (h - w) / 2
            right_pad = (h - w) / 2 + (h - w) % 2

            x1 = x1 - left_pad
            if x1 < 0:
                x1 = 0

            x2 = x2 + right_pad
            if x2 > img.shape[1]:
                x2 = img.shape[1]

        elif w > h:
            up_pad = (w - h) / 2
            down_pad = (w - h) / 2 + (w - h) % 2

            y1 = y1 - up_pad
            if y1 < 0:
                y1 = 0

            y2 = y2 + down_pad
            if y2 > img.shape[0]:
                y2 = img.shape[0]

        cvisib = visib[j]
        if y1 >= 0:
            cposey = posey[j] - y1
        else:
            cposey = posey[j] - box[1]

        if x1 >= 0:
            cposex = posex[j] - x1
        else:
            cposex = posex[j] - box[0]


        if y1 < 0:
            y1 = 0
        if x1 < 0:
            x1 = 0

        x1, x2, y1, y2 = np.int(x1), np.int(x2), np.int(y1), np.int(y2)
        patch = img[y1:y2, x1:x2]
        bboxes[j] = np.array([x1, y1, x2, y2])
        posey[j] = cposey
        posex[j] = cposex
        cv2.imwrite(vid_path + '/' + video + '/' + imgs[j].split('.')[0] + '_cropped.png', patch)

        ff['bbox'] = bboxes
        ff['y'] = posey
        ff['x'] = posex
        np.savez(ann_path + '/' + video + '.npz', **ff)
