import torch
from typing import Union, Tuple, List
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, BlendParams, Materials, SoftPhongShader, \
    look_at_view_transform, TexturesVertex, DirectionalLights, MeshRenderer
from .utils import Camera, BatchWrapper


class Renderer:
    @classmethod
    def render_rgb(cls,
                   vertices: torch.Tensor,
                   faces: torch.Tensor,
                   colors: torch.Tensor,
                   dist: Union[float, torch.Tensor, List[float]] = 10.0,
                   elev: Union[float, torch.Tensor, List[float]] = 0.0,
                   azim: Union[float, torch.Tensor, List[float]] = 0.0,
                   img_size: Union[int, Tuple[int, int], List[int]] = 256,
                   light_direction: Union[torch.Tensor, List[float], Tuple[float, float, float]] = None,
                   fov: float = 50,
                   znear: float = 0.01,
                   zfar: float = 100,
                   camera_type: str = 'orthographic',
                   device: Union[str, torch.device] = 'cuda'
                   ) -> torch.Tensor:

        batch_num = 1 if vertices.ndim == 2 or vertices.size(0) == 1 else len(vertices)

        vertices, faces, colors = BatchWrapper.make_batch_mesh(vertices, faces, colors)
        light_directions = BatchWrapper.make_batch_light_direction(light_direction, device, batch_num)
        dists, elevs, azims = BatchWrapper.make_batch_transformation(dist, elev, azim, device, batch_num)

        cameras = Camera.get_cameras(camera_type=camera_type, fov=fov, znear=znear, zfar=zfar, device=device)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(
            image_size=img_size, blur_radius=0.0, faces_per_pixel=1))

        blend_params = BlendParams(background_color=[0.0, 0.0, 0.0])
        materials = Materials(device=device)

        render_imgs = []

        for i in range(batch_num):
            R, T = look_at_view_transform(dist=dists[i], elev=elevs[i], azim=azims[i], device=device)

            textures = TexturesVertex(verts_features=colors[i][None])
            meshes = Meshes(verts=[vertices[i]], faces=[faces[i]], textures=textures)

            lights = DirectionalLights(direction=light_directions[i][None], device=device)
            shader = SoftPhongShader(cameras=cameras, lights=lights, blend_params=blend_params, materials=materials)
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

            render_imgs.append(renderer(meshes_world=meshes, R=R, T=T)[..., :3])

        return torch.cat(render_imgs)  # (B, H, W, C)
