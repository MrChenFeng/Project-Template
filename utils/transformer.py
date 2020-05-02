from skimage import io, transform
import numpy as np
import torch

'''
Transformations for landmarks labelled datasets.

Dataset should be dict{'image': imgdata, 'label': labeldata}
'''

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        h, w = image.shape[:2]

        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        img = torch.as_tensor(img)
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * torch.tensor([new_w / w, new_h / h])

        return {'image': img, 'label': landmarks, 'rotation':None}


class Crop(object):
    """Crop the image to a given bounding box.

    Args:
        output_box (tuple or int): Desired output box. If int, square crop
            is made.
    """

    def __init__(self, output_box):
            self.output_size = output_box

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        left = self.output_size[0]
        top = self.output_size[1]
        right = self.output_size[2]
        bottom = self.output_size[3]

        image = image[bottom: top, left: right]

        landmarks = landmarks - [left, top]

        return {'image': image, 'label': landmarks}

class Rotate(object):
    '''
    Rotate images by given degree. Samples should be save in dict
    '''
    def __call__(self, sample):
        image, landmarks, rot = sample['image'], sample['label'], sample['rotation']
        rotation = np.random.random_integers(0,3)
        rot = torch.zeros(4)
        rot[rotation] = 1
        for i in range(rotation):
            image = torch.rot90(image)
            # Error here. Rot will change the dimension!!!!!!!
            #landmarks = torch.rot90(landmarks)

        return {'image': image, 'label': landmarks, 'rotation':rot}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks, rot = sample['image'], sample['label'], sample['rotation']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.permute(2, 0, 1)
        return {'image': image.float(),
                'label': landmarks.float(),
                'rotation': rot.float()}
