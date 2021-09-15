from .base import BaseDataset
from . import utils
import torch
import numpy as np
import os.path as osp
import re
from PIL import Image

class Dataset300WLP(BaseDataset):
    
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.sub_dataset = ['AFW', 'AFW_Flip', 'IBUG', 'IBUG_Flip', 'HELEN', 'HELEN_Flip', 'LFPW', 'LFPW_Flip']

        for dataset in self.sub_dataset:
            with open(osp.join(self.root, 'filelist/', dataset + '_filelist.txt')) as f:
                file_names = f.read().splitlines()
            
            file_names = [name.replace(' ', '') for name in file_names]
            self.image_names += file_names
            file_names = [name.replace('_Flip', '') for name in file_names]
            self.obj_names += [re.sub(r'_[0-9]+.png', '.obj', name) for name in file_names]
            
            with open(osp.join(self.root, 'filelist/', dataset+'_param.dat')) as fd:
                paras = np.fromfile(file=fd, dtype=np.float32)
            self.m = np.append(self.m, paras.reshape((-1,294)).astype(np.float32)[:,1:9], axis=0)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Porcess image / mask
        image = Image.open(osp.join(self.root, 'image/', self.image_names[idx]))
        input_image = self.transform(image)
        image = self.image_loader(image)
        mask = self.image_loader(Image.open(osp.join(self.root, 'mask/', self.image_names[idx])))
        #canonical_image = self.image_loader(Image.open(osp.join(self.root, 'render_raw/', self.obj_names[idx].split('.')[0]+'.png')))
        
        m_i = utils.get_transform_matrix(torch.tensor(self.m[idx]))

        # Read processed obj file
        data = np.load(osp.join(self.root, 'process_raw/', self.obj_names[idx].split('.')[0]+'.npz'))
        vertices = torch.tensor(data['vertices'])
        scale = torch.tensor(data['scale'])
        shift = torch.tensor(data['shift'])[None]

        return {
                'data_name': self.image_names[idx].split('.')[0],
                'image':image.cuda(), 
                'input_image':input_image.cuda(),
                'mask':mask.cuda(), 
                'm':m_i.cuda(), 
                'vertices':vertices.cuda(), 
                'scale':scale.cuda(), 
                'shift':shift.cuda(),
                }