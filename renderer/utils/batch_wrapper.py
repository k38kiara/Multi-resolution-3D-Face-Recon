import torch
from typing import Union, Tuple, List


class BatchWrapper:
    @staticmethod
    def convert_to_tensor(data, device: Union[str, torch.device], add_one_dim=False, dtype=torch.float) -> torch.Tensor:
        result = torch.tensor(data, device=device, dtype=dtype)
        return result if not add_one_dim else result[None]

    @staticmethod
    def align_batch_data(data: torch.Tensor, batch_num: int):
        assert data.ndim == 2
        if batch_num > 1 and data.size(0) == 1:
            return torch.cat([data.clone() for _ in range(batch_num)])
        return data

    @classmethod
    def make_batch_transformation(cls,
                                  dist: Union[int, float, torch.Tensor, List[int], List[float]],
                                  elev: Union[int, float, torch.Tensor, List[int], List[float]],
                                  azim: Union[int, float, torch.Tensor, List[int], List[float]],
                                  device: Union[str, torch.device],
                                  batch_num: int):
        def make_batch_view(data, device):
            if isinstance(data, (int, float)):
                data = float(data)
                data = cls.convert_to_tensor(data, device, add_one_dim=True)[None]
            elif isinstance(data, torch.Tensor):
                if data.ndim == 0:
                    data = data[None]
                if data.ndim == 1:
                    data = data[None]
            elif isinstance(data, list):
                data = cls.convert_to_tensor(data, device).unsqueeze(1)
            else:
                raise TypeError('Cannot handle this type of transformation parameter -> "%s"' % type(data))
            return data

        dist = cls.align_batch_data(make_batch_view(dist, device), batch_num)
        elev = cls.align_batch_data(make_batch_view(elev, device), batch_num)
        azim = cls.align_batch_data(make_batch_view(azim, device), batch_num)
        return dist, elev, azim

    @classmethod
    def make_batch_color(cls,
                         light_direction: Union[torch.Tensor, List[float], Tuple[float, float, float]],
                         ambient_color: Union[torch.Tensor, List[float], Tuple[float, float, float]],
                         diffuse_color: Union[torch.Tensor, List[float], Tuple[float, float, float]],
                         specular_color: Union[torch.Tensor, List[float], Tuple[float, float, float]],
                         device: Union[str, torch.device],
                         batch_num: int):
        def make_batch(data, device):
            if isinstance(data, (list, tuple)):
                add_one_dim = not isinstance(data[0], (list, tuple))
                data = cls.convert_to_tensor(data, device, add_one_dim=add_one_dim)

            elif isinstance(data, torch.Tensor):
                data = data.float()
                if data.ndim == 1:
                    data = data[None]
            else:
                raise TypeError('Cannot handle this type of lighting parameter -> "%s"' % type(data))
            return data

        light_direction = cls.align_batch_data(make_batch(light_direction, device), batch_num)
        ambient_color = cls.align_batch_data(make_batch(ambient_color, device), batch_num)
        diffuse_color = cls.align_batch_data(make_batch(diffuse_color, device), batch_num)
        specular_color = cls.align_batch_data(make_batch(specular_color, device), batch_num)

        return light_direction, ambient_color, diffuse_color, specular_color

    @classmethod
    def make_batch_mesh(cls, vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor):
        if vertices.ndim == 2:
            vertices = vertices[None]
        if colors is None:
            colors = torch.ones_like(vertices)
        elif colors.ndim == 2:
            colors = colors[None]
        if faces.ndim == 2:
            faces = torch.cat([faces.clone()[None] for i in range(len(vertices))])

        return vertices, faces, colors
