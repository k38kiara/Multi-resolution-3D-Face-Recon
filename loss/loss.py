import torch
import torch.nn as nn
from typing import Dict
from .utils import HsvConverter, NormLoss, PerceptualLoss
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.loss.mesh_edge_loss import mesh_edge_loss
from pytorch3d.structures.meshes import Meshes

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual_model = PerceptualLoss().cuda()

    def get_edge_length_loss(self, vertices, faces):
        if len(faces.shape) < len(vertices.shape):
            faces = faces.repeat(vertices.shape[0], 1, 1)
        meshes = Meshes(verts=vertices, faces=faces)
        return mesh_edge_loss(meshes)

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

    def get_perceptual_loss(self, images: torch.Tensor, gt_images: torch.Tensor):
        return self.perceptual_model(images, gt_images)
    
    def get_model_loss(self, 
                output: Dict[str, torch.Tensor], 
                target: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:

        chamfer_dist, edge_length_loss = 0, 0
        for i in range(len(output['vertices'])):
            chamfer_dist = chamfer_dist + self.get_chamfer_dist(output['vertices'][i], target['vertices']) 
            edge_length_loss = edge_length_loss + self.get_edge_length_loss(output['vertices'][i], output['faces'][i])

        pixel_loss = self.get_pixel_loss(output['images'], target['images'], target['masks'])
        pixel_hsv_loss = self.get_pixel_hsv(output['images'], target['images'], target['masks'])
        symmetric_loss = self.get_symmetric_loss(output['canonical_images'])
        perceptual_loss = self.get_perceptual_loss(output['images'], target['images'])


        return {'chamfer_dist': chamfer_dist,
                'edge_length': edge_length_loss,
                'pixel': pixel_loss,
                'pixel_hsv': pixel_hsv_loss,
                'symmetric': symmetric_loss,
                'perceptual': perceptual_loss,
                }
