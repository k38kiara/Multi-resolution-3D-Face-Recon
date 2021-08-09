import torch
from typing import Union, List, Tuple
from .batch_wrapper import BatchWrapper

class Color:
    @classmethod
    def get_colors(cls,
                   light_direction: Union[torch.Tensor, List[float], Tuple[float, float, float]],
                   ambient_color: Union[torch.Tensor, List[float], Tuple[float, float, float]],
                   diffuse_color: Union[torch.Tensor, List[float], Tuple[float, float, float]],
                   specular_color: Union[torch.Tensor, List[float], Tuple[float, float, float]],
                   device: Union[str, torch.device],
                   batch_num: int
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if light_direction is None:
            light_direction = [0, 0, 1]
        if ambient_color is None:
            ambient_color = [1, 1, 1]
        if diffuse_color is None:
            diffuse_color = [1, 1, 1]
        if specular_color is None:
            specular_color = [0, 0, 0]

        light_direction, ambient_color, diffuse_color, specular_color = BatchWrapper.make_batch_color(
            light_direction, ambient_color, diffuse_color, specular_color, device, batch_num)

        return light_direction, ambient_color, diffuse_color, specular_color
