# Transformers
# - CentralCrop: Cropping image patches to given size
# - Resize: Resize image to square size by PIL ANTIALIAS operation
# - ImageToTensor: Convert to image tensor
# - RandomRotateImage: Rotate image patch by a random degree(90,180,270,360)
# - ConcatSequence: Convert list of Tensors to Tensor array
import torchvision.transforms as transforms
from PIL import Image
import torch
from .util_functions import rotate_img


class CentralCrop(object):
    """
    Center crop image patch.
    """

    def __init__(self, size):
        self.crop = transforms.CenterCrop(size)

    def __call__(self, input):
        if type(input) == list:
            return [self.crop(i) for i in input]
        return self.crop(input)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        if type(input) == list:
            return [i.resize(self.size, Image.ANTIALIAS) for i in input]
        return input.resize(self.size, Image.ANTIALIAS)


class ImageToTensor(object):
    """
    Converts a PIL image or sequence of PIL images into (a) PyTorch tensor(s).
    """

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, input):
        if type(input) == list:
            return [self.to_tensor(i) for i in input]
        return self.to_tensor(input)


class RandomRotateImage(object):
    """
    Important!!!
    Rotate image with a random degree and save corresponding label.
    """

    def __init__(self):
        pass

    def __call__(self, x):
        # if type(x) == list:
        #     label = []
        #     image = []
        #     for i in x:
        #         rot = torch.randint(0, 4, [1]).item()
        #         label.append(rot)
        #         image.append(rotate_img(i, rot))
        # else:
            # for each image, we will get a rotated one and original one
        rotated_label = torch.randint(1, 4, [1]).item()
        rotated_image = rotate_img(x, rotated_label)
        label = 0
        image = x
        # print(label)
        return {"image": image.clone().detach(), "label": torch.tensor(label),
                "rotated_image": rotated_image.clone().detach(), "rotated_label": torch.tensor(rotated_label)}


class ConcatSequence(object):
    """
    Concatenates a sequence (list of tensors) along a new axis.
    """

    def __init__(self):
        pass

    def __call__(self, input):
        return torch.stack(input)
