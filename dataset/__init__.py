from .dataset300WLP import Dataset300WLP
from .datasetAFLW2000 import DatasetAFLW2000
from . import utils
import torch
import os.path as osp
import numpy as np
from pytorch3d.structures import Meshes
from tqdm import tqdm
import os
from . import utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, root_path='/data/hank/Face/'):
        self.dataset = None
        if mode == '300W-LP':
            self.dataset = Dataset300WLP(root_path)
        elif mode == 'AFLW2000':
            self.dataset =  DatasetAFLW2000(root_path)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def preprocess_obj(obj_path, process_path):
        print('[INFO] Pre-processing object data...')

        for d in ['AFW', 'IBUG', 'HELEN', 'LFPW', 'AFLW2000']:
            files = os.listdir(osp.join(obj_path, d))
            for file in tqdm(files):
                vertices, _ = utils.get_obj_from_file(osp.join(obj_path, d, file), is_color=False)
                vertices, scale, shift = utils.get_normalized_vertices(vertices)
                file = file.split('.')[0] + '.npz'
                np.savez(osp.join(process_path, d, file), vertices=vertices, scale=scale, shift=shift)

    @staticmethod
    def load_mean_face(meanface_path):
        mean_faces, mean_vertices, mean_edges = [], [], []
        
        for i in ['505', '1961', '7726']:
            vertices, faces = utils.get_obj_from_file(osp.join(meanface_path, 'meanface_{}.obj'.format(i)))
            vertices, _, _ = utils.get_normalized_vertices(vertices)
            edges = utils.get_edges(vertices, faces)
            mean_faces.append(faces)
            mean_vertices.append(vertices)
            mean_edges.append(edges)

        return mean_faces, mean_vertices, mean_edges

    # @staticmethod
    # def 