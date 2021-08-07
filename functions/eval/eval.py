import torch
from tqdm import tqdm
from renderer import Renderer
from dataset import Dataset
from loss import Loss
from typing import Dict, Tuple

class Evaluator:

    @classmethod
    def run(cls, model, data_loader, mean_data, checkpoint) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = 0
        losses = {
                'chamfer_dist': 0,
                'pixel': 0,
                'pixel_hsv': 0,
                'symmetric': 0,
                }

        progress_bar_val = tqdm(data_loader, ascii=True)
        step_num = len(data_loader)
        
        if checkpoint is not None:
            model.load_state_dict(checkpoint)
        
        for data in progress_bar_val:
            step_loss, step_losses = cls.evaluate_step(model, data, mean_data)
            
            total_loss += step_loss
            for key in losses:
                losses[key] += step_losses[key]
        
        return total_loss / step_num, {key: losses[key] / step_num for key in losses}
    
    def evaluate_step(self, model, data, mean_data) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        with torch.no_grad():
            x3_shape, x2_shape, x1_shape, x3_color, pred_light = model(mean_data['vertices'],
                                                                        mean_data['edges'], 
                                                                        data['input_img'].float().cuda(), 
                                                                        data['m'].float().cuda())

            batch_size = x3_shape.shape[0]
            pred_vertices3 = x3_shape + mean_vertices[2].repeat(batch_size, 1, 1)
            pred_vertices2 = x2_shape + mean_vertices[1].repeat(batch_size, 1, 1)
            pred_vertices1 = x1_shape + mean_vertices[0].repeat(batch_size, 1, 1)

            output_images = Renderer.render_rgb(vertices=transform_vertices_coord(pred_vertices3, data),
                            faces=faces,
                            colors=colors,
                            light_direction=spheric2cartesian(pred_light),
                            )
            
            total_loss, losses = Loss.get_model_loss(output={'vertices': pred_vertices3, 'images': output_images},
                                                    target={'vertices': data['vertices'], 'images': data['image'], 'masks': data['mask']})

        return total_loss, losses



    @staticmethod
    def transform_vertices_coord(self, vertices, data):
        vertices = torch.mul(vertices, data['scale'][:, None, None].cuda())
        vertices = vertices +  data['shift'].cuda()
        vertices = torch.cat((vertices, torch.ones((b, vn, 1)).cuda()), -1)
        vertices = torch.matmul(vertices, torch.transpose(data['m'].cuda(), 1, 2))
        vertices = (vertices - 128) / 128 # Normalize to [-1, 1]
        
        return vertices

    @staticmethod
    def spheric2cartesian(light):
        r = 1
        theta, phi = light[:, 0] * 2 * math.pi, light[:, 1] * 2 * math.pi
        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi)
        light_dir = torch.cat((x[..., None], y[..., None], z[..., None]), axis=-1)
        return light_dir


