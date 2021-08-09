import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self._feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2),
                nn.LeakyReLU(negative_slope =0.2),
                
                nn.Conv2d(16, 32, 5, stride=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope =0.2),
                
                nn.Conv2d(32, 64, 5, stride=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope =0.2),
                
                nn.Conv2d(64, 128, 5, stride=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope =0.2),

                nn.Conv2d(128, 256, 5, stride=2),
                nn.BatchNorm2d(256),
                
                nn.LeakyReLU(),
                )
        self._fc = nn.Sequential(
            nn.Linear(6400, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope =0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
            )
 
    def forward(self, img):
        feature = self._feature_extractor(img)
        b = feature.shape[0]
        feature = feature.view(b, -1)
        pred = self._fc(feature)
        return pred