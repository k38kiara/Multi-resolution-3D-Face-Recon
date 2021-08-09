import torch
import torch.nn as nn

class LightDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 9)
        self.linear_rgb = torch.nn.Linear(512, 9)
        self.linear_dir = torch.nn.Linear(512, 2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, img_feature):
        b = img_feature.size(0)
        x = self.avgpool(img_feature)
        x = x.view(b, -1)
        # x = self.linear(x)


        pred_rgb = self.linear_rgb(x) #(512, 9)
        pred_rgb = self.sigmoid(pred_rgb)
        pred_dir = self.linear_dir(x) #(512, 3)
        pred_dir = self.sigmoid(pred_dir)


        return torch.cat((pred_rgb, pred_dir), -1) #(b, 9)