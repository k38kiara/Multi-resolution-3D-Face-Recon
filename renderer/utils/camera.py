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
                    device: Union[str, torch.device]
                    ) -> Union[FoVOrthographicCameras, FoVPerspectiveCameras]:

        if camera_type == 'orthographic':
            return cls.get_orthographic_cameras(znear, zfar, device)
        elif camera_type == 'perspective':
            return cls.get_perspective_cameras(fov, znear, zfar, device)
        else:
            raise ValueError('Cannot construct the camera which type = "%s"' % camera_type)

    @staticmethod
    def get_orthographic_cameras(znear: float,
                                 zfar: float,
                                 device: Union[str, torch.device]
                                 ) -> FoVOrthographicCameras:
        return FoVOrthographicCameras(znear=znear, zfar=zfar, device=device)

    @staticmethod
    def get_perspective_cameras(fov: float,
                                znear: float,
                                zfar: float,
                                device: Union[str, torch.device]
                                ) -> FoVPerspectiveCameras:
        return FoVPerspectiveCameras(fov=fov, znear=znear, zfar=zfar, device=device)
