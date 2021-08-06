from .rgb import RGBRenderer
from .depth import DepthRenderer


class Renderer:
    @staticmethod
    def render_rgb(*args, **kwargs):
        return RGBRenderer.render(*args, **kwargs)

    @staticmethod
    def render_depth(*args, **kwargs):
        return DepthRenderer.render(*args, **kwargs)
