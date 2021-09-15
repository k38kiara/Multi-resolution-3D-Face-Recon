import torch
import torch.nn as nn
from .layers import GProject, GBottleneck, GUnpooling, ResNet18, LightDecoder
from torch_geometric.nn import GCNConv

VERTEX_DIM = 3
GLOBAL_DIM = 128 #6
LOCAL_DIM = 256
GCN_OUTPUT_FEATURE = 128
MEANFACE_VERTEX_NUM = [505, 1961, 7726]

class Model(nn.Module):
    def __init__(self, mesh_number):
        super(Model, self).__init__()

        n_dim = 3
        self.K = mesh_number
        self.fnum = GLOBAL_DIM + LOCAL_DIM + LOCAL_DIM
        self.encoder = ResNet18(sample_vnum=MEANFACE_VERTEX_NUM[0], output_dim=GLOBAL_DIM)
        self.gcns = GBottleneck(in_dim=n_dim + GLOBAL_DIM + LOCAL_DIM, hidden_dim=64, out_dim=32)

        self.gcns_color = GBottleneck(in_dim=3 + GLOBAL_DIM + LOCAL_DIM, hidden_dim=64, out_dim=32)

        self.unpooling = nn.ModuleList([GUnpooling(0),
                                        GUnpooling(1)
                                    ])

        self.project = GProject()
        self.local_pool = torch.nn.AdaptiveMaxPool1d(LOCAL_DIM)
        self.light_decoder = LightDecoder()
    
    def forward(self, mean_vertices, edges, img, data):
        
        batch_size = img.size(0)
        global_features, local_features, encode_feat = self.encoder(img)

        # Shape
        # GCN Block 1
        zeros_padding = torch.zeros(batch_size, mean_vertices[0].shape[0], mean_vertices[0].shape[1]).cuda()
        x = []
        s = mean_vertices[0] + zeros_padding
        
        for k in range(self.K):
            local_feature = self.project(s, local_features, data, is_inverse=True)
            local_feature = self.local_pool(local_feature)
            x_input = torch.cat((s, local_feature, global_features), 2)
            x_output = self.gcns(x_input, edges[k])
            s = s + x_output
            x.append(s)
            
            if k < self.K - 1:
                s = self.unpooling[k](s)
                global_features = self.unpooling[k](global_features)

            elif k == self.K - 1:
                local_feature = self.project(s, local_features, data, is_inverse=True)
                local_feature = self.local_pool(local_feature)
                x_input = torch.cat((s, local_feature, global_features), 2)
                x3_color = self.gcns_color(x_input, edges[k])

        # Light Decoder
        pred_light = self.light_decoder(encode_feat)

        return x, x3_color, pred_light