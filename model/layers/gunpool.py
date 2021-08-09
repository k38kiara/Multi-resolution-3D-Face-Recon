import torch
import torch.nn as nn

class GUnpooling(nn.Module):
    """Graph Pooling layer, aims to add additional vertices to the graph.
    The middle point of each edges are added, and its feature is simply
    the average of the two edge vertices.
    Three middle points are connected in each triangle.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, unpool_data_path):
        # input: [batch, vnum, fnum]
        with open (unpool_data_path, 'rb') as fp:
            unpool_idx = pickle.load(fp)
        unpool_idx = torch.tensor(unpool_idx, dtype=torch.long)

        new_features = inputs[:, unpool_idx, :].clone()
        new_vertices = new_features.sum(2) / 2
        output = torch.cat([inputs, new_vertices], 1)

        return output