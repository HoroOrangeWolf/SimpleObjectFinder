import os

import numpy
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from scipy import misc
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 4])

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.img_labels.iloc[idx, 0:4]
        value = []
        for x in label:
            value.append(x)
        return image, numpy.array(value)
