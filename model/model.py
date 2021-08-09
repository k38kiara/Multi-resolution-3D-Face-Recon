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
    def __init__(self):
        super(Model, self).__init__()

        n_dim = 3
        self.fnum = GLOBAL_DIM + LOCAL_DIM + LOCAL_DIM
        self.encoder = ResNet18(sample_vnum=MEANFACE_VERTEX_NUM[0], output_dim=GLOBAL_DIM)
        self.gcns = nn.ModuleList([
            GBottleneck(in_dim=n_dim + GLOBAL_DIM + LOCAL_DIM, hidden_dim=128, out_dim=n_dim),
            GBottleneck(in_dim=n_dim + GLOBAL_DIM + LOCAL_DIM, hidden_dim=128, out_dim=n_dim),
            GBottleneck(in_dim=n_dim + GLOBAL_DIM + LOCAL_DIM, hidden_dim=128, out_dim=64),
            GCNConv(64, 3)
        ])
        
        self.gcns_color = nn.ModuleList([
            GBottleneck(in_dim=3 + GLOBAL_DIM + LOCAL_DIM, hidden_dim=256, out_dim=64),
            GCNConv(64, 3)
        ])
        
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
        local_feature = self.project(mean_vertices[0]+zeros_padding, local_features, data, is_inverse=True) # [batch, h, vnum, fnum]
        local_feature = self.local_pool(local_feature)
        x = torch.cat((mean_vertices[0].repeat(batch_size, 1, 1), local_feature, global_features), 2) # [batch, vnum, fnum]
        x1, x_hidden = self.gcns[0](x, edges[0]) # [batch, vnum, 3], [batch, vnum, 128]
        
        # GCN Block 2
        local_feature = self.project(mean_vertices[0]+x1, local_features, data, is_inverse=True)
        local_feature = self.local_pool(local_feature)

        x = self.unpooling[0](torch.cat((x1, local_feature, global_features), 2))
        x2, x_hidden = self.gcns[1](x, edges[1])
        
        # GCN Block 3
        local_feature = self.project(mean_vertices[1]+x2, local_features, data, is_inverse=True)
        local_feature = self.local_pool(local_feature)
        global_features = self.unpooling[0](global_features)
        x = self.unpooling[1](torch.cat((x2, local_feature, global_features), 2))
        x3, x_hidden = self.gcns[2](x, edges[2])
        x3 = self.gcns[3](x3, edges[2])

        # Texture GCN
        pred_vertex_pos = x3 + mean_vertices[2]
        pred_vertex_pos = torch.clamp(pred_vertex_pos, 0, 1)

        local_feature = self.project(mean_vertices[2]+x3, local_features, data, is_inverse=True)
        local_feature = self.local_pool(local_feature)
        global_features = self.unpooling[1](global_features)


        x = torch.cat((local_feature, global_features, pred_vertex_pos), 2)
        x3_color, _ = self.gcns_color[0](x, edges[2])
        x3_color = self.gcns_color[1](x3_color, edges[2])

        # Light Decoder
        pred_light = self.light_decoder(encode_feat)

        return x3, x2, x1, x3_color, pred_light