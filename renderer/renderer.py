import torch
from torch import nn
import math
import pytorch3d.renderer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.cameras import OpenGLOrthographicCameras
from pytorch3d.renderer import RasterizationSettings, look_at_view_transform
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer import MeshRasterizer
from pytorch3d.renderer import HardPhongShader, HardGouraudShader, SoftGouraudShader, BlendParams
from pytorch3d.renderer import PointLights, DirectionalLights

class pytorch3d_render():
    def __init__(self, render_size=256):
        dist = 1000
        elev = 0
        azim = 0
        _R, _T = look_at_view_transform(dist, elev, azim)
        image_size = render_size

        # Initialize an OpenGL perspective camera.
        cameras = OpenGLOrthographicCameras(
            znear=-450,
            zfar=1000.0,
            R=_R,
            T=_T,
            top=render_size/2,
            bottom=-render_size/2,
            left=-render_size/2,
            right=render_size/2,
            device=torch.device('cuda'),
        )

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters.
        raster_settings = RasterizationSettings(image_size=image_size)
        blend_params = BlendParams(background_color=[0.0, 0.0, 0.0])

        # Create a phong renderer by composing a rasterizer and a shader. Here we can use a predefined
        # PhongShader, passing in the device on which to initialize the default parameters
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardGouraudShader(device=torch.device('cuda'), cameras=cameras, blend_params=blend_params))


    def render(self, vertices, colors, faces, light=None, is_directory=False, shade=False):
        
        textures = TexturesVertex(verts_features=colors)
        batch = vertices.shape[0]
        
        if len(faces.shape) < 3:
            faces = faces.repeat(batch, 1, 1)

        faces = self.reverse_back_face(vertices, faces)
        pred_mesh = Meshes(vertices, faces, textures)

        if light is None:
            ambient = torch.tensor([1,1,1]).repeat(batch, 1).float()*0.5
            diffuse = torch.tensor([1,1,1]).repeat(batch, 1).float()*0.5
            specular = torch.tensor([0,0,0]).repeat(batch, 1).float()
            light_dir = torch.tensor([0,0,1]).repeat(batch, 1).float()
        else:
            ambient = torch.tensor([1,1,1]).repeat(batch, 1).float()*0.5
            diffuse = torch.tensor([1,1,1]).repeat(batch, 1).float()*0.5
            specular = torch.tensor([0,0,0]).repeat(batch, 1).float()
            light_dir = self.spheric2cartesian(light)

        lights = DirectionalLights(ambient_color=ambient, 
                                   diffuse_color=diffuse, 
                                   specular_color=specular, 
                                   direction=light_dir,
                                   device='cuda')

        img = self.renderer(pred_mesh, lights=lights)
        img = torch.clamp(img, 0.0, 1.0)
        return img[..., :3]

    def spheric2cartesian(self, light):
        r = 1
        theta, phi = light[:, 0] * 2 * math.pi, light[:, 1] * 2 * math.pi
        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi)
        light_dir = torch.cat((x[..., None], y[..., None], z[..., None]), axis=-1)
        return light_dir

    def get_normal_vectors(self, vertices, faces):
        # vertices = [b, n, 3], faces = [b, n, 3]
        B = vertices.size(0)
        batch_indices = torch.arange(B).view(-1, 1)

        vec1 = vertices[batch_indices, faces[..., 1]] - vertices[batch_indices, faces[..., 0]]
        vec2 = vertices[batch_indices, faces[..., 2]] - vertices[batch_indices, faces[..., 1]]
        
        return torch.cross(vec1, vec2, dim=2)
            

    def reverse_back_face(self, vertices, faces):
        r_faces = faces.clone()

        normals = self.get_normal_vectors(vertices, faces)
        indices = normals[..., 2] < 0

        r_faces[indices] = torch.index_select(r_faces[indices], 1, torch.tensor([0, 2, 1], dtype=torch.long, device=faces.device))

        return r_faces