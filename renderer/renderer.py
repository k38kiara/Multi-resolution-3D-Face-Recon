from .rgb import RGBRenderer
from .depth import DepthRenderer


class Renderer:
    @staticmethod
    def render_rgb(*args, **kwargs):
        RGBRenderer.render(*args, **kwargs)

    @staticmethod
    def render_depth(*args, **kwargs):
        DepthRenderer.render(*args, **kwargs)
