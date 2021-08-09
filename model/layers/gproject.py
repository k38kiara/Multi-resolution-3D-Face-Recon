import torch
import torch.nn as nn
import torch.nn.functional as F

class GProject(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, vertices, img_feats, deform, rotation, is_inverse=False):
        '''
        img_feat: [batch, cnum, h, w]
        inputs: [batch, h, vnum, 2], in range[-1, 1]
        output: [batch, vnum, feat_dim(cnum)]
        '''
        output = []
        sample_point = vertices + deform
        sample_point = sample_point.detach()
        sample_point_inverse = sample_point.clone() * torch.tensor([-1, 1, 1]).cuda()

        sample_point = torch.matmul(sample_point, torch.transpose(rotation, 1, 2))[..., 0:2].unsqueeze(1)
        sample_point_inverse = torch.matmul(sample_point_inverse, torch.transpose(rotation, 1, 2))[..., 0:2].unsqueeze(1)

        sample_point[..., 1] = sample_point[..., 1] * -1
        sample_point = sample_point[..., [1, 0]]
        sample_point_inverse[..., 1] = sample_point_inverse[..., 1] * -1
        sample_point_inverse = sample_point_inverse[..., [1, 0]]
        
        for feat in img_feats:
            sample_feature = F.grid_sample(feat, sample_point)
            if is_inverse:
                sample_feature = (sample_feature + F.grid_sample(feat, sample_point_inverse)) / 2
            output.append(torch.transpose(sample_feature.squeeze(2), 1, 2))
        output = torch.cat(output, 2)

        return output