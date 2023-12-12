from .blender import *
from .llff import *


dataset_dict = {
    "blender": SinglescaleBlenderDataset,
    "multiscale_blender": MultiscaleBlenderDataset,
    "video_blender": VideoBlenderDataset,
    "llff": SinglescaleLLFFDataset,
    "multiscale_llff": MultiscaleLLFFDataset
}
