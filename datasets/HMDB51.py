import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os


class HMDB51(Dataset):
    '''
    No keypoints label
    '''

    def __init__(self, root_path, resize, num_per_pair=6):
        self.__clip_len = int(num_per_pair / 2)
        self.root_path = root_path
        self.resize = resize
        self.__count_videos()

    def __getitem__(self, index):
        action_id = np.abs(self.__videos_count - index)
        sample_id = np.argmin(action_id)
        video_id = action_id[sample_id]
        action_list = os.listdir(self.root_path)
        sample_path = os.path.join(self.root_path, action_list[sample_id])
        sample_list = os.listdir(sample_path)
        video_path = os.path.join(sample_path, sample_list[video_id])
        frames_list = sorted(os.listdir(video_path))
        start_id = np.random.randint(0, len(frames_list) - 2)
        # end_id = start_id + 2
        image_data = []

        for i in range(self.__clip_len):
            i_path = os.path.join(video_path, frames_list[start_id + i])
            image = Image.open(i_path)
            image = image.resize(self.resize, Image.ANTIALIAS)
            image_data.append(np.asarray(image))

        sample = np.asarray(image_data)

        return sample

    def __count_videos(self):
        count = np.array([], dtype=np.int)
        order = 0
        for i in os.listdir(self.root_path):
            count = np.append(count, 0)
            if order > 0:
                count[order] = len(os.listdir(os.path.join(self.root_path, i))) + count[order - 1]
            else:
                count[order] = len(os.listdir(os.path.join(self.root_path, i)))
            order += 1
        self.__videos_count = count

    def __len__(self):
        return self.__videos_count[-1]


if __name__ == '__main__':
    # Calling the function
    test = HMDB51(root_path='/home/chen/Video_Disentanled_Representation/Datasets/HMDB51Frames', resize=(128, 128))
    t1 = len(test)
    t = test[100]
