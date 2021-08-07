from .base import BaseDataset
from . import utils
import torch
import numpy as np
import os.path as osp
import re
from PIL import Image

class DatasetAFLW2000(BaseDataset):
    
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.sub_dataset = ['AFLW2000']

        for dataset in self.sub_dataset:
            with open(osp.join(self.root, 'filelist/', dataset + '_filelist.txt')) as f:
                file_names = f.read().splitlines()
            
            file_names = [name.replace(' ', '') for name in file_names]
            self.image_names += file_names
            self.obj_names += [re.sub(r'png', 'obj', name) for name in file_names]
            
            with open(osp.join(self.root, 'filelist/', dataset+'_param.dat')) as fd:
                paras = np.fromfile(file=fd, dtype=np.float32)
            self.m = np.append(self.m, paras.reshape((-1,294)).astype(np.float32)[:,1:9], axis=0)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Porcess image / mask
        image = self.image_loader(Image.open(osp.join(self.root, 'image/', self.image_names[idx])))
        input_image = self.transform(image)
        image = self.image_loader(image)
        mask = self.image_loader(Image.open(osp.join(self.root, 'mask/', self.image_names[idx])))
        m_i = utils.get_transform_matrix(torch.tensor(self.m[idx]))

        # Read processed obj file
        data = np.load(osp.join(self.root, 'process/', self.obj_names[idx].split('.')[0]+'.npz'))
        vertices = torch.tensor(data['vertices'])
        scale = torch.tensor(data['scale'])
        shift = torch.tensor(data['shift'])

        return {
                'image':image, 
                'input_image':input_image, 
                'mask':mask, 
                'm':m_i, 
                'vertices':vertices, 
                'scale':scale, 
                'shift':shift,
                }