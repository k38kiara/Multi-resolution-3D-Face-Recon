from .base import BaseDataset
from . import utils
import torch
import numpy as np
import os.path as osp
import re
from PIL import Image

class DatasetCustomer(BaseDataset):
    
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.image_names = ['./nasic_test/nasic02.png']
        self.mask_names = ['./nasic_test/nasic02_mask.png']

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Porcess image / mask

        image = Image.open(self.image_names[idx])
        input_image = self.transform(image)
        image = self.image_loader(image)
        mask = self.image_loader(Image.open(self.mask_names[idx]))
        m_i = torch.tensor([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])

        # Read processed obj file
        vertices = None
        scale = torch.tensor(120)
        shift = torch.tensor([128, 98, 128])[None]

        return {
                'data_name': 'customer/'+self.image_names[idx].split('.')[0].split('/')[-1],
                'image':image.cuda(), 
                'input_image':input_image.cuda(), 
                'mask':mask.cuda(), 
                'm':m_i.cuda(), 
                'scale':scale.cuda(), 
                'shift':shift.cuda(),
                'pose':torch.ones(3),
                }