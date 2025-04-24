from .mlp import MLP
from .smallVGG import SmallVGG
from .movileNetV1 import MobileNetV1
from .vit import ViT
from .fft_vit import FFTViT
from .custom_vit import CustomViT
from .dynamic_range_vit import DynamicRangeViT


try:
    from .custom_model import *
except:
    pass