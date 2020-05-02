import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
import torch
from transformers import Rotate

ImageFile.LOAD_TRUNCATED_IMAGES = True


class WFLW(Dataset):
    """
    WFLW
    
    It should be noted, rotation will be converted finally.
    """

    def __init__(self, dataroot='/home/chen/Datasets/Faces and human pose/WFLW/WFLW_images', is_rotated = False, is_train=True, transform=None, size= None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = '/home/chen/Datasets/Faces and human pose/WFLW/Processed_annotations/trainlist.csv'
        else:
            self.csv_file = '/home/chen/Datasets/Faces and human pose/WFLW/Processed_annotations/testlist.csv'

        self.is_train = is_train
        self.transform = transform
        self.data_root = dataroot
        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)
        self.pad = 5
        if is_rotated:
            self.rotation = Rotate() 
        self.is_rotated = is_rotated
#         if size is not None,
        self.size = size
#             choices = np.random.permutation(len(self.landmarks_frame))[size]
#             self.landmarks_frame

        
    def __len__(self):
        if self.size is not None:
            return self.size
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.landmarks_frame.iloc[idx, 0])
        
        left = self.landmarks_frame.iloc[idx, 1]
        top = self.landmarks_frame.iloc[idx, 2]
        right = self.landmarks_frame.iloc[idx, 3]
        bottom = self.landmarks_frame.iloc[idx, 4]
        
        #bbox = torch.tensor([top,left,bottom,right])
        pts = self.landmarks_frame.iloc[idx, 5:].values
        pts = pts.astype('float').reshape(-1, 2)
        
        right = np.maximum(np.max(pts[:,0]), right) + self.pad
        left = np.minimum(np.min(pts[:,0]), left) -self.pad
        bottom = np.maximum(np.max(pts[:,1]), bottom) + self.pad
        top = np.minimum(np.min(pts[:,1]), top) - self.pad
        
        img = Image.open(image_path).convert("RGB")

        img = img.crop([left, top, right, bottom])
        landmarks = pts - [left,top]
#         print(pts[0])
#         print(landmarks[0])
        sample = {'image': img, 'landmarks': landmarks}
        if self.transform is not None:
            sample = self.transform(sample)
        # meta = {
        #     "index": idx,
        #     "center": center,
        #     "rescale": scale,
        #     "pts": torch.Tensor(pts),
        #     # "tpts": tpts,
        #     "bbox": bbox,
        # }
        if self.is_rotated:
            sample['rotations'] = None
            sample = self.rotation(sample)
            
        return sample


if __name__ == '__main__':
    from transformers import Rescale, Normalize, ToTensor
    from torchvision.transforms import Compose

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t = WFLW(transform=Compose([Rescale((224, 224)), ToTensor(), Normalize(mean, std)]))
    tmp = t[1000]
    import matplotlib.pyplot as plt

    img = tmp['image']
    landmark = tmp['landmarks']


    def display(image, label):
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        plt.imshow(image)
        plt.scatter(label[:, 0], label[:, 1], c='r', marker='o')
        # plt.scatter(bbox[:, 0], bbox[:, 1], c='r', marker='o')
        plt.show()

    display(img, landmark)
