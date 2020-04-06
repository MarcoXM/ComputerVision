import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import pandas as pd 
import joblib
import numpy as np 
import torch
from albumentations.augmentations import functional as aF
from PIL import Image

class ThreeDatasetTrain:
    def __init__(self,images,labels, transform = None):
        
        self.labels = labels
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform!= None:
            image,label = self.transform(image,label)
        image = image-1 ## normalize to -1 and 1 
        label = label-1
        return torch.tensor(image,dtype=torch.float),torch.tensor(label,dtype=torch.float)