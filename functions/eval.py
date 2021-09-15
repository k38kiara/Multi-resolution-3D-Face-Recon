from . import utils
import torch
from dataset import Dataset
from torch.utils.data import DataLoader
from model import Model, Discriminator
from renderer import pytorch3d_render
from loss import Model_Loss
from tqdm import tqdm
import config
from tensorboard import Tensorboard

Renderer = pytorch3d_render()

class Evaluator:

    def __init__(self, args):
        
        self.dataset = Dataset('AFLW2000')
        self.dataloader = DataLoader(self.dataset, batch_size=8, num_workers=0, collate_fn=Dataset.collate_func)
        self.mean_faces, self.mean_vertices, self.mean_edges, self.gt_faces = Dataset.load_mean_face(config.DATASET_ROOT + 'meandata/')
        self.criteria = Model_Loss()
        self.tensorboard = Tensorboard(args.ckpt)

    def _reset_loss(self):
        return {'chamfer_dist': 0.,
                'laplacian_smooth': 0,
                'landmarks_loss': 0.,
                'pixel': 0.,
                'symmetric': 0.,
                'perceptual': 0.,
                }

    def run(self, model, epoch):
        loss_item = self._reset_loss()
        progress_bar_eval = tqdm(self.dataloader, ascii=True)
        for i, data in enumerate(progress_bar_eval):
            loss, align_images, canonical_images = self._eval_step(model, data)
            loss_item = {key : loss_item[key] + loss[key] for key in loss_item}
            progress_bar_eval.set_description('Loss: %.6f' %(loss['chamfer_dist']))

        loss_item = {key : loss_item[key] / len(progress_bar_eval) for key in loss_item}
        self.tensorboard.write_data(data={'loss':loss_item, 
                                        'pred_images':torch.clamp(align_images, 0, 1), 
                                        'gt_images':data['image'], 
                                        'conanical_images': torch.clamp(canonical_images, 0, 1)},
                                         epoch=epoch, 
                                         mode='eval')
        return loss_item


    def _eval_step(self, model, data):
        batch_size = len(data['input_image'])
        
        pred_vertices, color, pred_light = model(self.mean_vertices, self.mean_edges, data['input_image'].float(), data)

        rendered_images = Renderer.render(vertices=utils.transform_vertices_coord(pred_vertices[2], data), 
                                            faces=self.mean_faces[2],
                                            colors=(color+1)/2,  
                                            light=pred_light
                                           )
        rendered_canonical_images = Renderer.render(vertices=pred_vertices[2]*128, 
                                            faces=self.mean_faces[2],
                                            colors=(color+1)/2
                                           )

        mask = torch.where(rendered_images.detach().cpu() == torch.tensor([0, 0, 0]), torch.tensor(0), torch.tensor(1)).cuda()
        mask = mask * data['mask']
        align_images = torch.mul(rendered_images, mask) + torch.mul(data['image'], 1-mask)
        loss = self.criteria.get_model_loss(
                                    output={'vertices': pred_vertices,
                                            'faces': [self.mean_faces[i] for i in range(3)],
                                            'images': align_images,
                                            'canonical_images': rendered_canonical_images,},
                                    target={'vertices': torch.stack(data['vertices']),
                                            'faces': self.gt_faces,
                                            'images': data['image'],
                                            'masks': data['mask'],
                                            'mean_vertices': self.mean_vertices}
                                    )

        return loss, align_images, rendered_canonical_images


