import torch
import torch.nn as nn
import torch.nn.functional as F

class GProject(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vertices, img_feats, data, is_inverse=False):
        '''
        img_feat: [batch, cnum, h, w]
        inputs: [batch, h, vnum, 2], in range[-1, 1]
        output: [batch, vnum, feat_dim(cnum)]
        '''
        output = []
        sample_point = vertices.detach()
        sample_point_inverse = sample_point.clone() * torch.tensor([-1, 1, 1]).cuda()


        sample_point = self.transform_vertices(sample_point, data)[..., 0:2].unsqueeze(1)
        sample_point_inverse = self.transform_vertices(sample_point_inverse, data)[..., 0:2].unsqueeze(1)


        # sample_point[..., 1] = sample_point[..., 1] * -1
        # sample_point = sample_point[..., [1, 0]]
        # sample_point_inverse[..., 1] = sample_point_inverse[..., 1] * -1
        # sample_point_inverse = sample_point_inverse[..., [1, 0]]
        
        for feat in img_feats:
            sample_feature = F.grid_sample(feat, sample_point)
            if is_inverse:
                sample_feature = (sample_feature + F.grid_sample(feat, sample_point_inverse)) / 2
            output.append(torch.transpose(sample_feature.squeeze(2), 1, 2))
        output = torch.cat(output, 2)

        return output


    def transform_vertices(self, vertices, data):


        b, vn, c = vertices.shape

        vertices = torch.mul(vertices, data['scale'][:, None, None].cuda())
        vertices = vertices +  data['shift'].cuda()
        
        vertices = torch.cat((vertices, torch.ones((b, vn, 1)).cuda()), -1)
        vertices = torch.matmul(vertices, torch.transpose(data['m'].cuda(), 1, 2))
        vertices[..., 0:2] = (vertices[..., 0:2] - 128) / 128

        return vertices
