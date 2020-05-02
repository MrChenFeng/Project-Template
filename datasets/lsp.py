import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
from scipy.io import loadmat
import torch

class LSP(Dataset):
    '''
    Already cropped.

    ~/images: original images
    ~/visualized: with pose estimations
    ~/joints: keypoints
    '''

    def __init__(self, root_path, resize):
        self.root_path = root_path
        self.resize = resize
        self._num = len([f for f in os.listdir(root_path+'/images') if f.endswith('jpg')])

        self.label_path = os.path.join(os.path.abspath(root_path), 'joints.mat')
        self.frame_path = os.path.join(os.path.abspath(root_path), 'images')

        ### joints.mat
        self.labels = loadmat(self.label_path)

        ### frames list
        self.frames = sorted([f for f in os.listdir(self.frame_path) if f.endswith('.jpg')])

    def __getitem__(self, index):
        frame_path = os.path.join(self.frame_path, self.frames[index])
        label = self.labels['joints'][:,:,index]
        # end_id = start_id + 2
        image = Image.open(frame_path)
        image = image.resize(self.resize, Image.ANTIALIAS)

        image = np.asarray(image)
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        #label =

        sample = {'image': image/torch.max(image),
                'label': label}

        return sample

    def __len__(self):
        return self._num


if __name__ == '__main__':
    # Calling the function
    test = LSP(root_path='/home/chen/Video_Disentanled_Representation/Datasets/LeedsSport', resize=(128, 128))
    t1 = len(test)
    t = test[100]