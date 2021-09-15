import torch
import torch.nn as nn

class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points1: torch.Tensor, points2: torch.Tensor, w1=1.0, w2=1.0, each_batch=False):
        self.check_parameters(points1)
        self.check_parameters(points2)

        sample_points1 = points1[:, torch.randint(high=points1.shape[1], size=(min(points1.shape[1], 2048),))]
        sample_points2 = points2[:, torch.randint(high=points2.shape[1], size=(2048,))]


        diff = sample_points1[:, :, None, :] - sample_points2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist1 = dist
        dist2 = torch.transpose(dist, 1, 2)

        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)

        dist_min1, indices1 = torch.min(dist1, dim=2)
        dist_min2, indices2 = torch.min(dist2, dim=2)

        loss1 = dist_min1.mean(1)
        loss2 = dist_min2.mean(1)

        loss = w1 * loss1 + w2 * loss2

        if not each_batch:
            loss = loss.mean()

        return loss#, indices1, indices2

    @staticmethod
    def check_parameters(points: torch.Tensor):
        assert points.ndimension() == 3  # (B, N, 3)
        assert points.size(-1) == 3