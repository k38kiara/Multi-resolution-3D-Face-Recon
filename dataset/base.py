from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

class BaseDataset(Dataset):
    def __init__(self):
        # self.image_loader = transforms.Compose([
        #                     transforms.ToTensor()])
        
        self.transform = transforms.Compose([
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])

        self.m = np.empty([0,8])
        self.image_names = []
        self.obj_names = []

    def image_loader(self, image):
        return torch.tensor(np.array(image) / 255.)


