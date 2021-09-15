from . import utils
import os
from . import eval
import torch
from torch.optim import Adam, SGD
from dataset import Dataset
from torch.utils.data import DataLoader
from model import Model, Discriminator
from renderer import pytorch3d_render
from loss import Model_Loss
from tensorboard import Tensorboard
from tqdm import tqdm
import config
import numpy as np
import pickle

Renderer = pytorch3d_render()

class Refiner():
    def __init__(self, args):
        
        self.dataset = Dataset(args.dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=1, num_workers=0, collate_fn=Dataset.collate_func, shuffle=False)
        self.mean_faces, self.mean_vertices, self.mean_edges, self.gt_faces = Dataset.load_mean_face(config.DATASET_ROOT + 'meandata/')
        
        self.model = Model().cuda()
        self.checkpoint = torch.load(config.CKPT_ROOT + '{}/last_model.pt'.format(args.ckpt))
        #self.optimizer = Adam(params=self.model.parameters(), lr=1e-5, weight_decay=1e-5, betas=(0.5, 0.9))
        self.optimizer = Adam(params=[
            {'params':self.model.gcns_color.parameters(), 'lr':1e-4,},
            {'params':self.model.light_decoder.parameters(), 'lr':1e-5,},
            ], weight_decay=1e-5, betas=(0.5, 0.9))
        #self.D = Discriminator().cuda()

        # self.weight = {'chamfer_dist': 10, 
        #                'edge_length': 1,
        #                'laplacian_smooth': 10,
        #                'landmarks_loss': 1,
        #                'pixel': 100, 
        #                'pixel_hsv': 1, 
        #                'symmetric': 0.1, 
        #                'perceptual': 0.1, 
        #                }
        self.weight = {'chamfer_dist': 10, 
                       'laplacian_smooth': 1,
                       'landmarks_loss': 1,
                       'pixel': 10, 
                       'symmetric': 0.1, 
                       'perceptual': 0.1,
                       #'canonical': 1,
                       #'pixel_hsv': 1,
                       #'edge_length': 1, 
                       }
        self.criteria = Model_Loss()
        self.finetune_epoch = args.epoch
        self.save_path = config.OUTPUT_ROOT + args.output + '/'
        os.makedirs(self.save_path+'img/', exist_ok=True)
        os.makedirs(self.save_path+'obj/', exist_ok=True)
        os.makedirs(self.save_path+'light/', exist_ok=True)

    def _reset_loss(self):
        return {'chamfer_dist': 0.,
                'laplacian_smooth': 0,
                'landmarks_loss': 0.,
                'pixel': 0.,
                'symmetric': 0.,
                'perceptual': 0.,
                #'canonical': 0,
                #'pixel_hsv': 0.,
                #'edge_length': 0.,
                }

    def save_data(self, pred_data, data):
        pred_vertices, colors, pred_light, align_images, align_shades = pred_data
        file_name = data['data_name'][0].split('/')[1]
        for i in range(len(pred_vertices)):
            utils.save_obj(vertices=pred_vertices[i][0].detach().cpu().numpy(), 
                    colors=(colors[0, :len(self.mean_vertices[i])]+1)/2,
                    faces=self.mean_faces[i], 
                    save_path=self.save_path + 'obj/' + file_name + '_{}.obj'.format(i))
        utils.save_imgs([data['image'][0].detach().cpu().numpy(), 
                        align_images[0].detach().cpu().numpy(),
                        align_shades[0].detach().cpu().numpy(),], 
                        save_path=self.save_path + 'img/' + file_name + '.png')
        np.savez(self.save_path + 'light/' + file_name + '.npz', light=pred_light.detach().cpu().numpy())

    # def save_high_res(self, s_high_res, t_high_res):
    #     utils.save_obj(vertices=s_high_res[0].detach().cpu().numpy(), 
    #                 colors=(t_high_res[0]+1)/2,
    #                 faces=self.mean_faces[-1], 
    #                 save_path=self.save_path + 'obj/' + 'test' + '_high.obj')


    def run(self):
        for i, data in enumerate(self.dataloader):
            if i > 10:
                break
            self.model.train()
            self.model.load_state_dict(self.checkpoint)
            
            min_loss = 100.
            best_result = None
            
            progress_bar_finetune = tqdm(range(self.finetune_epoch), ascii=True)
            for epoch in progress_bar_finetune:
                loss, pred_data = self._step(data, epoch)
                progress_bar_finetune.set_description('Loss: %.6f' %(loss))
                
                if loss < min_loss:
                    min_loss = loss
                    best_result = pred_data
            self.save_data(best_result, data)

    # def _run_high_res(self, data, s_k):
    #     def _last_unpool(inputs):
    #         with open ('/data/hank/Face/meandata/unpool_idx_2.dat', 'rb') as fp:
    #             unpool_idx = pickle.load(fp)
    #         unpool_idx = torch.tensor(unpool_idx, dtype=torch.long)
    #         new_features = inputs[:, unpool_idx, :].clone()
    #         new_vertices = new_features.sum(2) / 2
    #         output = torch.cat([inputs, new_vertices], 1)
    #         return output
        
    #     s_k = _last_unpool(s_k)
    #     global_features, local_features, encode_feat = self.model.encoder(data['input_image'].float())
    #     for i in range(2):
    #         global_features = self.model.unpooling[i](global_features)
    #     global_features = _last_unpool(global_features)

    #     local_feature = self.model.project(s_k, local_features, data, is_inverse=True)
    #     local_feature = self.model.local_pool(local_feature)
    #     x_input = torch.cat((s_k, local_feature, global_features), 2)
    #     x_output = self.model.gcns(x_input, self.mean_edges[-1])
    #     s_high_res = self.mean_vertices[-1] + x_output
    #     x_input = torch.cat((s_high_res, local_feature, global_features), 2)
    #     t_high_res = self.model.gcns_color(x_input, self.mean_edges[-1])
    #     return s_high_res, t_high_res


    def _step(self, data, epoch):
        batch_size = len(data['input_image'])
        pred_vertices, color, pred_light = self.model(self.mean_vertices, self.mean_edges, data['input_image'].float(), data)
        transformed_vertices = utils.transform_vertices_coord(pred_vertices[2], data)
        rendered_images = Renderer.render(vertices=transformed_vertices, 
                                            faces=self.mean_faces[2],
                                            colors=(color+1)/2,  
                                            light=pred_light)

        rendered_canonical_images = Renderer.render(vertices=pred_vertices[2]*128, 
                                            faces=self.mean_faces[2],
                                            colors=(color+1)/2
                                           )
        align_shades = Renderer.render(vertices=pred_vertices[2]*128, 
                                            faces=self.mean_faces[2],
                                            colors=(color+1)*0+1
                                           )


        mask = torch.where(rendered_images.detach().cpu() == torch.tensor([0, 0, 0]), torch.tensor(0), torch.tensor(1)).cuda()
        align_mask = mask * data['mask']
        align_images = torch.mul(rendered_images, align_mask) + torch.mul(data['image'], 1-align_mask)

        loss_item = {'pixel': self.criteria.get_pixel_loss(align_images, data['image'], data['mask']),
                    'symmetric': self.criteria.get_symmetric_loss(rendered_canonical_images),
                    'perceptual': self.criteria.get_perceptual_loss(align_images, data['image'])}

        loss = sum([loss_item[key] * self.weight[key] for key in loss_item])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, [pred_vertices, color, pred_light, align_images, align_shades]