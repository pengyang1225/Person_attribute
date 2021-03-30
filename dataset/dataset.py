"""
author: guopei
"""
import os
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from conf import settings

class Person_Attribute_Train(Dataset):

    def __init__(self, path, transform=None):

        self.root = path
        self.transform = transform
        with open(os.path.join(self.root, 'train.txt')) as f:
            imgs = f.readlines()
        imgs = [img.rstrip("\n") for img in imgs]
        random.shuffle(imgs)
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        sample = self.imgs[index]
        words = sample.split(" ")
        img_path = words[0]
        #labels = [int(i) for i in words[1:]]
        labels = [i for i in words[1:]]
        labels = np.int32(labels)
        image = cv2.imread(os.path.join(settings.IMAGE_PATH,img_path))

        if self.transform:
            image = self.transform(image)

        return image, labels



class Person_Attribute_Test(Dataset):

    def __init__(self, path, transform=None):

        self.root = path
        self.transform = transform
        with open(os.path.join(self.root, 'test.txt')) as f:
            imgs = f.readlines()
        imgs = [img.rstrip("\n") for img in imgs]
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        sample = self.imgs[index]
        words = sample.split(" ")
        img_path = words[0]
        labels = [int(i) for i in words[1:]]
        labels = np.int32(labels)
        image = cv2.imread(os.path.join(settings.IMAGE_PATH,img_path))

        if self.transform:
            image = self.transform(image)

        return image, labels

