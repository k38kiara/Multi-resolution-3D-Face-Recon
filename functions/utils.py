import torch
import numpy as np
import math
from PIL import Image

def transform_vertices_coord(vertices, data):
    b, vn, c = vertices.shape
    vertices = torch.mul(vertices, data['scale'][:, None, None].cuda())
    vertices = vertices +  data['shift'].cuda()
    vertices = torch.cat((vertices, torch.ones((b, vn, 1)).cuda()), -1)
    vertices = torch.matmul(vertices, torch.transpose(data['m'], 1, 2))
    vertices = (vertices - 128)
    vertices[..., 2] = vertices[..., 2] / 1e4
    
    return vertices

def spheric2cartesian(light):
    r = 1
    theta, phi = light[:, 0] * 2 * math.pi, light[:, 1] * 2 * math.pi
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    light_dir = torch.cat((x[..., None], y[..., None], z[..., None]), axis=-1)
    return light_dir

def save_obj(vertices, colors, faces, save_path):
    
    f_out = open(save_path, 'w')
    for v, c in zip(vertices, colors):
        f_out.write('v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], c[0], c[1], c[2]))
    for i in faces:
        f_out.write('f {} {} {}\n'.format(int(i[0])+1, int(i[1])+1, int(i[2])+1))

def save_imgs(imgs, save_path):
    n_size = len(imgs)
    full_images = np.ones((256, (256 + 10) * n_size - 10, 3))
    for i, img in enumerate(imgs):
        full_images[0:256, i*(256+10) : i*(256+10)+256] = img
    full_images = (full_images*255).astype('uint8')

    im = Image.fromarray(full_images)
    im.save(save_path)

from kaolin.graphics import DIBRenderer, NeuralMeshRenderer
def kaolin_render(vertices, faces, colors, azim=90, elev=0, dist=2, render_size=256):
    batch = len(vertices)
    # Color shape = [batch, vnum, 3]
    renderer = DIBRenderer(render_size, render_size)
    renderer.set_look_at_parameters([azim for i in range(batch)], [elev for i in range(batch)], [dist for i in range(batch)])
    render_img, _, _ = renderer.forward(points=[vertices, faces], colors_bxpx3=colors)

    return render_img   # [1, h, w, c]