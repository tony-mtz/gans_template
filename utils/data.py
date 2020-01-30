import torch
import numpy as np

class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, images, train=True, labels=None):

        self.images = images
        self.train = train
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
        image preprocessing and reshape before
        getitem
        '''
        image = self.images[idx]
        image = image.reshape((1,28,28)).astype(np.float32)
        if self.labels:
            label = 1
            return (image, label)
        return image