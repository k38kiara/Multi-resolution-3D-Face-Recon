import numpy as np
import re
import torch
from pytorch3d.structures import Meshes

def get_obj_from_file(file_path, reverse_face=False, is_color=False):
    with open(file_path, 'r') as f:
        raw_data = f.read()
    
    vertices = np.array(re.findall(r'v (.+) (.+) (.+) (.+) (.+) (.+)', raw_data), dtype=np.float)
    faces = np.array(re.findall(r'f (.+) (.+) (.+)', raw_data), dtype=np.int) - 1
    
    if reverse_face:
        tmp = faces[:, 2].copy()
        faces[:, 2] = faces[:, 0].copy()
        faces[:, 0] = tmp
    if not is_color:
        vertices = vertices[:, :3]

    return torch.tensor(vertices), torch.tensor(faces)

def get_normalized_vertices(vertices: torch.Tensor):
    shift = vertices[:, :3].mean(0)
    vertices[:, :3] -= shift
    #scale = np.linalg.norm(vertices[:, :3], axis=1).max()
    scale = torch.norm(vertices[:, :3], dim=1).max()
    vertices[:, :3] /= scale

    return vertices, scale, shift

def get_edges(vertices: torch.Tensor, faces: torch.Tensor):
    mesh = Meshes(vertices[:, :3][None], faces[None].long())
    edges = mesh.edges_packed()
    return torch.cat([edges, edges.flip([1])]).transpose(0, 1)

def get_transform_matrix(m: torch.Tensor):

    m_i = torch.transpose(torch.reshape(m, (4, 2)), 0, 1)
    m_i_row1 = torch.nn.functional.normalize(m_i[0,0:3], dim=0, p=2)
    m_i_row2 = torch.nn.functional.normalize(m_i[1,0:3], dim=0, p=2)
    m_i_row3 = torch.cat((torch.reshape(torch.cross(m_i_row1, m_i_row2), (1, 3)), torch.zeros((1, 1)).double()), 1)
    m_i = torch.cat((m_i, m_i_row3), 0)
    return m_i