import torch
import torch.nn as nn
from typing import Dict
from .utils import HsvConverter, NormLoss
from pytorch3d.loss.chamfer import chamfer_distance

class Loss(nn.Module):
    def __init__(self, weight: Dict[str, float]):
        super().__init__()
        self.weight = weight

    def get_chamfer_dist(self, output: torch.Tensor, target: torch.Tensor):
        return chamfer_distance(output, target)[0]
    
    def get_laplacian(self, output, target):
        pass
    
    def get_symmetric_loss(self, images: torch.Tensor, loss_type='l1'):
        return NormLoss.norm(images, torch.flip(images, [2]), loss_type=loss_type)
    
    def get_pixel_loss(self, images: torch.Tensor, gt_images: torch.Tensor, masks: torch.Tensor, loss_type='l21'):
        return NormLoss.norm(images, gt_images, masks=masks, loss_type=loss_type)
    
    def get_pixel_hsv(self, images: torch.Tensor, gt_images: torch.Tensor, masks: torch.Tensor, loss_type='l21'):
        
        hsv_images = HsvConverter.rgb_to_hsv(images.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        hsv_gt_images = HsvConverter.rgb_to_hsv(gt_images.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return NormLoss.norm(hsv_images, hsv_gt_images, masks=masks, loss_type=loss_type)

    @classmethod
    def get_model_loss(cls, 
                output: Dict[str, torch.Tensor], 
                target: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        
        chamfer_dist = cls.get_chamfer_dist(output['vertices_3'], target['vertices'])
        if 'vertices_2' in output and 'vertices_1' in output
            chamfer_dist = chamfer_dist + cls.get_chamfer_dist(output['vertices_2'], target['vertices']) 
            chamfer_dist = chamfer_dist + cls.get_chamfer_dist(output['vertices_1'], target['vertices'])

        pixel_loss = cls.get_pixel_loss(output['images'], target['images'], target['masks'])
        pixel_hsv_loss = cls.get_pixel_hsv(output['images'], target['images'], target['masks'])
        symmetric_loss = cls.get_symmetric_loss(output['images'])

        total_loss = cls.weight['chamfer_dist'] * chamfer_dist \
                    + cls.weight['pixel'] * pixel_loss \
                    + cls.weight['pixel_hsv'] * pixel_hsv_loss \
                    + cls.weight['symmetric'] * symmetric_loss

        return total_loss, {
                'chamfer_dist': chamfer_dist,
                'pixel': pixel_loss,
                'pixel_hsv': pixel_hsv_loss,
                'symmetric': symmetric_loss,
                }
