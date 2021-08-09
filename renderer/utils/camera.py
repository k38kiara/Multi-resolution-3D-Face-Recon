import torch
from typing import Union
from pytorch3d.renderer import FoVOrthographicCameras, FoVPerspectiveCameras


class Camera:
    @classmethod
    def get_cameras(cls,
                    camera_type: str,
                    fov: float,
                    znear: float,
                    zfar: float,
                    device: Union[str, torch.device],
                    R: Union[torch.Tensor, None] = None,
                    T: Union[torch.Tensor, None] = None
                    ) -> Union[FoVOrthographicCameras, FoVPerspectiveCameras]:

        if camera_type == 'orthographic':
            return cls.get_orthographic_cameras(znear, zfar, R, T, device)
        elif camera_type == 'perspective':
            return cls.get_perspective_cameras(fov, znear, zfar, R, T, device)
        else:
            raise ValueError('Cannot construct the camera which type = "%s"' % camera_type)

    @staticmethod
    def get_orthographic_cameras(znear: float,
                                 zfar: float,
                                 R: Union[torch.Tensor, None],
                                 T: Union[torch.Tensor, None],
                                 device: Union[str, torch.device]
                                 ) -> FoVOrthographicCameras:
        return FoVOrthographicCameras(znear=znear, zfar=zfar, R=R, T=T, device=device)

    @staticmethod
    def get_perspective_cameras(fov: float,
                                znear: float,
                                zfar: float,
                                R: Union[torch.Tensor, None],
                                T: Union[torch.Tensor, None],
                                device: Union[str, torch.device]
                                ) -> FoVPerspectiveCameras:
        return FoVPerspectiveCameras(fov=fov, znear=znear, zfar=zfar, R=R, T=T,  device=device)
