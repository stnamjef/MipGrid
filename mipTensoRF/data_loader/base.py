import math
import numpy as np
import torch
from PIL import Image


def get_radius(directions):
    # radius, we use unnormlized directions: (H*W, 1)
    dx = torch.sqrt(torch.sum((directions[:-1, ...] - directions[1:, ...])**2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    # cut the distnace in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radius = dx[..., None] * 2 / math.sqrt(12)
    return radius


def get_radius_ndc(directions):
    # distance from each direction vector to its x-axis neighbor.
    dx = torch.sqrt(torch.sum((directions[:-1, :, :] - directions[1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    # distance from each direction vector to its y-axis neightbor.
    dy = torch.sqrt(torch.sum((directions[:, :-1, :] - directions[:, 1:, :])**2, -1))
    dy = torch.cat([dy, dy[:, -2:-1]], 1)
    # cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radius = (0.5 * (dx + dy))[..., None] * 2 / math.sqrt(12)
    return radius


def resize_and_normalize(img, width, height, downsample, flatten):
    # downsample
    if downsample > 1.0:
        img_downsampled = img.resize([width, height], Image.BILINEAR)
    else:
        img_downsampled = img
    # PIL Image to numpy array (H, W, 4)
    img = np.array(img_downsampled, dtype=np.float32) / 255.
    # blend A to RGB
    if img.shape[-1] > 3:
        img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
    # numpy to tensor
    img = torch.from_numpy(img)
    # (H*W, 3)
    if flatten:
        img = img.view(-1, 3)
    return img_downsampled, img


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def batch_indices(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


class BaseDataset:
    def __init__(
        self,
        data_dir, 
        split, 
        batch_size, 
        n_downsamples=4,
        downsample=1.0, 
        is_stack=False, 
        n_vis=-1, 
        return_radii=False, 
        patch_size=1,
        device="cpu"
    ):
        if is_stack:
            assert batch_size == 1
        assert patch_size in [1, 2, 4]

        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.n_downsamples = n_downsamples
        self.downsample = downsample
        self.is_stack = is_stack
        self.n_vis = n_vis
        self.return_radii = return_radii
        self.patch_size = patch_size
        self.device = device

        self._load_renderings()

        if not is_stack:
            self.reset_sampler()
    
    def _load_renderings(self):
        raise NotImplementedError
    
    @property
    def n_scales(self):
        return self.n_downsamples
    
    @property
    def n_images(self):
        assert self.is_stack == True
        assert len(self.rays) % self.n_downsamples == 0
        return len(self.rays) // self.n_downsamples
    
    def reset_sampler(self):
        assert self.is_stack != True
        self.sampler = SimpleSampler(len(self.rays), self.batch_size)
    
    @torch.no_grad()
    def fetch_data(self, index=None):
        raise NotImplementedError
