import torch
from typing import Union, Tuple, List
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, BlendParams, look_at_view_transform
from .utils import Camera, BatchWrapper


class DepthRenderer:
    @classmethod
    def render(cls,
               vertices: torch.Tensor,
               faces: torch.Tensor,
               dist: Union[float, torch.Tensor, List[float]] = 10.0,
               elev: Union[float, torch.Tensor, List[float]] = 0.0,
               azim: Union[float, torch.Tensor, List[float]] = 0.0,
               img_size: Union[int, Tuple[int, int], List[int]] = 256,
               fov: float = 50,
               znear: float = 0.01,
               zfar: float = 100,
               camera_type: str = 'orthographic',
               is_normalize: bool = True,
               device: Union[str, torch.device] = 'cuda') -> torch.Tensor:

        batch_num = 1 if vertices.ndim == 2 or vertices.size(0) == 1 else len(vertices)

        vertices, faces, _ = BatchWrapper.make_batch_mesh(vertices, faces, None)
        meshes = Meshes(verts=vertices, faces=faces)

        dists, elevs, azims = BatchWrapper.make_batch_transformation(dist, elev, azim, device, batch_num)
        R, T = look_at_view_transform(dist=dists, elev=elevs, azim=azims, device=device)
        cameras = Camera.get_cameras(camera_type=camera_type, fov=fov, znear=znear, zfar=zfar, R=R, T=T, device=device)

        raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        rendered_depths = rasterizer(meshes).zbuf

        if is_normalize:
            for i, rendered_depth in enumerate(rendered_depths):
                rendered_depths[i] = cls.normalize_depth(rendered_depth)
        return rendered_depths  # (B, H, W, 1)

    @staticmethod
    def normalize_depth(depth: torch.Tensor):
        assert depth.ndimension() == 3
        assert depth.size(2) == 1

        depth_indices = depth >= 0
        non_depth_indices = depth < 0

        depth[depth_indices] = depth[depth_indices].max() - depth[depth_indices]
        depth[depth_indices] /= depth[depth_indices].max()
        depth[non_depth_indices] = torch.zeros_like(depth[non_depth_indices])

        return depth
