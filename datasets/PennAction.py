from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
# from scipy.io import loadmat
from torchvision import transforms

class PennAction(Dataset):
    '''
    Generated samples will be saved in a Dict.

    Keys:
    image : image arry
    label: keypoints x and y
    rotation: image rotation degree(1-hot code)
    '''
    def __init__(self, root_path, transform=None, num_per_pair=2, frame_interval=2):
        self._clip_len = int(num_per_pair)
        self._interval = int(frame_interval)

        self.transform = transform

        self.label_path = os.path.join(os.path.abspath(root_path), 'labels')
        self.video_path = os.path.join(os.path.abspath(root_path), 'frames')

        self.videos = sorted(os.listdir(self.video_path))
        self.labels = sorted([f for f in os.listdir(self.label_path) if f.endswith('.npz')])
        # self.resize = resize

        self.num = len(os.listdir(self.video_path))

    def __getitem__(self, index):
        selected_videos = self.videos[index]
        selected_labels = self.labels[index]
        selected_label_path = os.path.join(os.path.abspath(self.label_path), selected_labels)
        selected_video_path = os.path.join(os.path.abspath(self.video_path), selected_videos)
        frame_list = sorted([f for f in os.listdir(selected_video_path) if f.endswith('.png')])
        id = np.random.randint(0, len(frame_list) - self._interval * self._clip_len + 1)
        image = []
        label = []
        rotation = []
        for i in range(self._clip_len):
            tmp = self._get_one_frame(selected_label_path, selected_video_path, frame_list, id + i * self._interval)
            image.append(tmp['image'])
            label.append(tmp['label'])
            rotation.append(tmp['rotation'])

        image = torch.stack(image)
        label = torch.stack(label)
        rotation = torch.stack(rotation)
        # we will save rotation degree as label here
        sample = {'image':image,'label':label,'rotation': rotation}
        return sample

    def _get_one_frame(self, label_path, frame_path, frame_list, id):
        points = np.load(label_path)
        x_points = points['x']
        y_points = points['y']

        #vis = points['visibility'][id].astype(np.bool)
        # print(vis)
        # bounding box:
        # bbox = points['bbox']

        i_path = os.path.join(frame_path, frame_list[id])
        image = Image.open(i_path)
        image = torch.as_tensor(np.asarray(image))
        labelx = x_points[id]  # [vis]
        labely = y_points[id]  # [vis]
        # left = np.int(bbox[id][0])
        # top = np.int(bbox[id][1])
        # right = np.int(np.ceil(bbox[id][2]))
        # bottom = np.int(np.ceil(bbox[id][3]))
       # print(labelx.shape)
        label = torch.as_tensor([labelx, labely]).T
       # print(label.shape)
        # image = image[top:bottom, left: right]
        # label = label - [left, top]

        tmp = {'image': image, 'label': label, 'rotation':None}
        # image = image.resize(self.resize, Image.ANTIALIAS)
        if self.transform:
            tmp = self.transform(tmp)
        return tmp

    def __len__(self):
        return self.num


if __name__ == '__main__':
    # Calling the function
    import matplotlib.pyplot as plt
    from utils.transformer import *

    test = PennAction(root_path='/home/chen/Video_Disentanled_Representation/Datasets/Penn_Action',
                      transform=transforms.Compose([Rescale((192,192)),Rotate()]))
    t1 = len(test)
    plt.figure(figsize=(20,20))
    for i, k in enumerate(np.random.randint(2200, size=4)):
        t = test[k]
        for j in range(3):
            tt1 = t['image'][j]
           # tt2 = t['label'][0]
            plt.subplot(4, 3, i*3 + j +1)
            plt.imshow(tt1)

        #plt.scatter(tt2[:, 0], tt2[:, 1], c='r', marker='+')

    # forget the labels now

    plt.show()
