import torch
import torch.nn as nn
from typing import Dict
from .utils import HsvConverter, NormLoss, PerceptualLoss
# from pytorch3d.loss.chamfer import chamfer_distance as chamfer_distance
# from pytorch3d.loss.mesh_edge_loss import mesh_edge_loss
from kaolin.metrics.mesh import laplacian_loss
from pytorch3d.loss import mesh_laplacian_smoothing
from kaolin.metrics.mesh import chamfer_distance as mesh_cd
from .utils import ChamferDistanceLoss
from kaolin.metrics.mesh import edge_length
from pytorch3d.structures.meshes import Meshes
from kaolin.rep import TriangleMesh
import numpy as np
import re

class Model_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual_model = PerceptualLoss().cuda()
        self.bce_loss_func = nn.BCELoss()
        self.cd = ChamferDistanceLoss()
        self.landmarks = [np.load('../landmark_505.npz')['landmark'],
                          np.load('../landmark_1961.npz')['landmark'],
                          np.load('../landmark_7726.npz')['landmark']]

    def get_landmark_loss(self, vertices, target_vertices, landmarks_idx):
        return NormLoss.norm(vertices[:, landmarks_idx[:, 1]], 
                            target_vertices[:, landmarks_idx[:, 0]], 
                            loss_type='l2')

    def get_edge_length_loss(self, vertices, faces):
        edge_loss = 0.0
        for v in vertices:
            pre_mesh = TriangleMesh.from_tensors(vertices=v.float(), faces=faces.long())
            pre_mesh.cuda()
            edge_loss += edge_length(pre_mesh)
        return edge_loss / len(vertices)

    def get_chamfer_dist(self, vertices: torch.Tensor, target_vertices: torch.Tensor, faces, target_faces):
        cd_loss = 0.0
        for v, gt_v in zip(vertices, target_vertices):
            pre_mesh = TriangleMesh.from_tensors(vertices=v.float(), faces=faces.long())
            gt_mesh = TriangleMesh.from_tensors(vertices=gt_v.float(), faces=target_faces.long())
            pre_mesh.cuda()
            gt_mesh.cuda()
            cd_loss += mesh_cd(pre_mesh, gt_mesh, w1=1, w2=0.55)
            
            # sample_v = v[torch.randint(high=v.shape[0], size=(min(v.shape[0], 3000),))]
            # sample_gt_v = gt_v[torch.randint(high=gt_v.shape[0], size=(3000,))]
            # cd_loss += point_cd(sample_v, sample_gt_v, w1=1, w2=0.55)
        return cd_loss / len(vertices)
        #return self.cd(vertices, target_vertices, w1=1, w2=0.55)
    
    def get_laplacian(self, vertices: torch.Tensor, mean_mesh, faces):
        # if len(faces.shape) < len(vertices.shape):
        #     faces = faces.repeat(vertices.shape[0], 1, 1)
        # meshes = Meshes(verts=vertices, faces=faces)
        # return mesh_laplacian_smoothing(meshes)
        lap_loss = 0.0
        for v in vertices:
            pre_mesh = TriangleMesh.from_tensors(vertices=v.float(), faces=faces.long())
            pre_mesh.cuda()
            lap_loss += laplacian_loss(mean_mesh, pre_mesh)
        return lap_loss / len(vertices)

        
    
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
    
    def get_D_loss(self, input_result, is_real):
        batch_size = input_result.shape[0]
        if is_real:
            return self.bce_loss_func(input_result, torch.zeros(batch_size, 1).cuda())
        else:
            return self.bce_loss_func(input_result, torch.ones(batch_size, 1).cuda())

    def get_model_loss(self, 
                output: Dict[str, torch.Tensor], 
                target: Dict[str, torch.Tensor],
                ) -> Dict[str, torch.Tensor]:

        chamfer_dist, edge_length_loss, laplacian_smooth_loss, landmarks_loss = 0, 0, 0, 0
        for i in range(len(output['vertices'])):
            chamfer_dist = chamfer_dist + self.get_chamfer_dist(output['vertices'][i], target['vertices'], output['faces'][i], target['faces']) 
            mean_mesh = TriangleMesh.from_tensors(vertices=target['mean_vertices'][i].float(), faces=output['faces'][i].long())
            mean_mesh.cuda()
            laplacian_smooth_loss = laplacian_smooth_loss + self.get_laplacian(output['vertices'][i], mean_mesh, output['faces'][i])
            landmarks_loss = landmarks_loss + self.get_landmark_loss(output['vertices'][i], target['vertices'], self.landmarks[i])

        pixel_loss = self.get_pixel_loss(output['images'], target['images'], target['masks'])
        symmetric_loss = self.get_symmetric_loss(output['canonical_images'])
        perceptual_loss = self.get_perceptual_loss(output['images'], target['images'])


        return {'chamfer_dist': chamfer_dist,
                'laplacian_smooth': laplacian_smooth_loss,
                'landmarks_loss': landmarks_loss,
                'pixel': pixel_loss,
                'symmetric': symmetric_loss,
                'perceptual': perceptual_loss,
                } 