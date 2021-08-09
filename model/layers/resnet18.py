import torch
import torch.nn as nn
from torchvision.models import resnet18



class ResNet18(nn.Module):
    def __init__(self, sample_vnum, output_dim):
        self.output_dim = output_dim
        self.vertice_num = sample_vnum
        super().__init__()
        self._model = resnet18(pretrained=True) 
        self.linear = torch.nn.Linear(512, self.vertice_num * self.output_dim)

    def __make_layer(self, block, planes, blocks, stride=1, dilate=False):
        res = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        return res

    def forward(self, x):
        b = x.size(0)

        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        local_features = []

        x = self._model.layer1(x)
        y = torch.sum(y[0], axis=2)
        local_features.append(x)

        x = self._model.layer2(x)
        y = torch.sum(y[0], axis=2)
        local_features.append(x)

        x = self._model.layer3(x)
        y = torch.sum(y[0], axis=2)
        local_features.append(x)

        x = self._model.layer4(x)
        y = torch.sum(y[0], axis=2)
        local_features.append(x)

        g_features = self._model.avgpool(x) # 512
        g_features = g_features.view(b, -1)
        gs_features = self.linear(g_features)
        gs_features = gs_features.view(b, self.vertice_num, self.output_dim)

        return gs_features, local_features, x