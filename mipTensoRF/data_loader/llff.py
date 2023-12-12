import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from .ray_utils import *
from .base import BaseDataset, get_radius, resize_and_normalize


def normalize(v):
        """Normalize a vector."""
        return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


class LLFFDataset(BaseDataset):
    def __init__(self, hold_every, **kwargs):
        self.hold_every = hold_every
        self.white_bg = False
        self.near_far = [0.0, 1.0]
        self.aabb = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        
        # OpenGL (x: right, y: up, z: backward)
        # OpenCV (x: right, y: down, z: forward)
        self.blender2opencv = np.eye(4)  # this will change nothing

        super(LLFFDataset, self).__init__(**kwargs)

    def _load_llff(self):
        # read dataset
        pose_bounds = np.load(os.path.join(self.data_dir, "poses_bounds.npy"))  # (n_images, 17)
        img_paths = sorted(glob.glob(os.path.join(self.data_dir, "images_4/*")))

        # check if # images and poses are the same
        assert len(pose_bounds) == len(img_paths)

        # split array
        poses = pose_bounds[:, :15].reshape(-1, 3, 5)  # (n_images, 3, 5)
        nears, fars = np.hsplit(pose_bounds[:, -2:], 2)  # (n_images, 2)
        assert np.mean(nears < fars) == 1.0

        # step 1.1: rescale focal length to match that of images_4 (4 times downsampled)
        height, width, focal = poses[0, :, -1]  # original intrinsics, same for all images
        height, width, focal = int(height/4.0), int(width/4.0), focal/4.0

        # step 1.2: rescale focal length to match that of images_4 downsampled
        height, width, focal = int(height/self.downsample), int(width/self.downsample), focal/self.downsample

        # step 2: correct posses (to OpenGL system??)
        # original poses has rotation in form "down right back", change to "right up back"
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], axis=-1)
        poses, _ = center_poses(poses, self.blender2opencv)

        # step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        scale_factor = nears.min() * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1 / 0.75 = 1.33
        # nears /= scale_factor
        # fars /= scale_factor
        poses[..., 3] /= scale_factor

        return img_paths, poses, height, width, focal


class SinglescaleLLFFDataset(LLFFDataset):
    def _load_renderings(self):
        # sanity check
        assert self.n_downsamples == 1 and self.patch_size == 1

        # load llff dataset (4-times downsampled) 
        img_paths, poses, height, width, focal = self._load_llff()

        # ray directions for all pixels, same for all images
        directions = get_ray_directions_blender(height, width, [focal, focal])

        # radius at each pixel
        if self.return_radii:
            radii = []
            radius = get_radius(directions)
        else:
            radii = None

        # split train and test
        test_indices = np.arange(0, len(img_paths), self.hold_every)
        indices = test_indices if self.split != "train" else list(set(np.arange(len(img_paths))) - set(test_indices))

        # main loop
        rays, rgbs = [], []
        for i in tqdm(indices, position=0, leave=True, desc=f"{self.split:4s} data"):
            # camera pose in OpenGL system
            c2w = torch.tensor(poses[i], dtype=torch.float32)

            # load and preprocess image
            rgb = Image.open(img_paths[i])
            rgbs += [resize_and_normalize(rgb, width, height, self.downsample, not self.is_stack)[-1]]

            # rays in ndc space: (H*W, 6)
            ray_o, ray_d = get_rays(directions, c2w)
            ray_o, ray_d = ndc_rays_blender(height, width, focal, 1., ray_o, ray_d)
            rays += [torch.cat([ray_o, ray_d], dim=1)]  # (H*W, 6)

            # radius: (H*W, 1)
            if self.return_radii:
                radii += [radius.view(-1, 1)]
        
        # concatenate
        if not self.is_stack:
            self.rays = torch.cat(rays, dim=0)
            self.rgbs = torch.cat(rgbs, dim=0)          # (N*H*W, 3)
            if self.return_radii:
                self.radii = torch.cat(radii, dim=0)    # (N*H*W, 1)
            else:
                self.radii = radii                      # (1,) --> None
        else:
            self.rays = rays                            # (N, H*W, 3)
            self.rgbs = rgbs                            # (N, H, W, 3)
            self.radii = radii                          # (N, H*W, 1) or (1,)
        
        self.lossmults = None

    @torch.no_grad()
    def fetch_data(self, index=None):
        if not self.is_stack:
            # random indices w.r.t rays (training mode)
            # index = torch.randint(0, len(self.rays), (self.batch_size,))
            index = self.sampler.batch_indices()
        else:
            assert index != None

        # sample batch
        rays_batch = self.rays[index].to(self.device, non_blocking=True)
        rgbs_batch = self.rgbs[index].to(self.device, non_blocking=True)
        if self.return_radii:
            radii_batch = self.radii[index].to(self.device, non_blocking=True)
        else:
            radii_batch = None
        
        # the last None is for the lossmults
        return rays_batch, rgbs_batch, radii_batch, None
    

class MultiscaleLLFFDataset(LLFFDataset):
    def _load_renderings(self):
        # sanity check
        assert self.n_downsamples == 4 and self.downsample == 1.0
        assert self.return_radii == True

        # load llff dataset (4-times downsampled) 
        img_paths, poses, height, width, focal = self._load_llff()

        # width and height of each patch
        pW = pH = self.patch_size

        # split train and test
        test_indices = np.arange(0, len(img_paths), self.hold_every)
        indices = test_indices if self.split != "train" else list(set(np.arange(len(img_paths))) - set(test_indices))

        # main loop
        rays, rgbs, radii, lossmults = [], [], [], []
        for i in tqdm(indices, position=0, leave=True, desc=f"{self.split:4s} data"):
            # camera pose in OpenGL system
            c2w = torch.tensor(poses[i], dtype=torch.float32)

            # load and preprocess image
            rgb = Image.open(img_paths[i])

            # downsample to 4 different scales
            for j in range(self.n_downsamples):
                # rescale width and height
                iW, iH = width // (2**j), height // (2**j)

                # rescale, normalize, blend alpha: (ImgH, ImgW, 3) or (ImgH*ImgW, 3)
                rgb, _rgb = resize_and_normalize(rgb, iW, iH, 2**j, not self.is_stack)

                # rescale focal length
                fx, fy = focal / (2**j) * pW, focal / (2**j) * pH

                # ray direction from origin to the center of each pixel: (ImgH*PatchH, ImgW*PatchW, 3)
                direction = get_ray_directions_blender(iH * pH, iW * pW, [fy, fx])

                # ray origins and directions (not normalized): (ImgH*PatchH*ImgW*PatchW, 6)
                ray_o, ray_d = get_rays(direction, c2w)
                ray_o, ray_d = ndc_rays_blender(iH * pH, iW * pW, fx, 1., ray_o, ray_d)
                ray = torch.cat([ray_o, ray_d], dim=1)

                # radius: (ImgH*PatchH*ImgW*PatchW, 1)
                radius = get_radius(direction).view(-1, 1)

                # lossmultiplier: (H*W, 1)
                lossmult = torch.tensor([[4**j,]], dtype=torch.float32).repeat(iH*iW, 1)

                # reshape: (ImgH*PatchH*ImgW*PatchW, 6) --> (ImgH*ImgW, PatchH*PatchW, 6)
                if pH > 1 and pW > 1:
                    ray = ray.view(iH, pH, iW, pW, 6).transpose(1, 2).reshape(iH*iW, pH*pW, 6).contiguous()
                    radius = radius.view(iH, pH, iW, pW, 1).transpose(1, 2).reshape(iH*iW, pH*pW, 1).contiguous()
                
                # append items
                rays.append(ray)
                rgbs.append(_rgb)
                radii.append(radius)
                lossmults.append(lossmult)
        
        # concatenate
        if not self.is_stack:
            self.rays = torch.cat(rays, dim=0)              # (N*H*W, 3) or (N*iH*iW, pH*pW, 3)
            self.rgbs = torch.cat(rgbs, dim=0)              # (N*H*W, 3)
            self.radii = torch.cat(radii, dim=0)            # (N*H*W, 1) or (N*iH*iW, pH*pW, 1)
            self.lossmults = torch.cat(lossmults, dim=0)    # (N*H*W, 1)
        else:
            # reorder items: [800, ..., 400, ..., 200, ..., 100, ...]
            self.rays = [rays[j] for i in range(self.n_downsamples) for j in range(i, len(rays), self.n_downsamples)]                   # (N, H*W, 3) or (N, iH*iW, pH*pW, 3)
            self.rgbs = [rgbs[j] for i in range(self.n_downsamples) for j in range(i, len(rgbs), self.n_downsamples)]                   # (N, H, W, 3)
            self.radii = [radii[j] for i in range(self.n_downsamples) for j in range(i, len(radii), self.n_downsamples)]                # (N, H*W, 1) or (N, iH*iW, pH*pW, 1)
            self.lossmults = [lossmults[j] for i in range(self.n_downsamples) for j in range(i, len(lossmults), self.n_downsamples)]    # (N, H*W, 1)

    @torch.no_grad()
    def fetch_data(self, index=None):
        if not self.is_stack:
            # random indices w.r.t rays (training mode)
            # index = torch.randint(0, len(self.rays), (self.batch_size,))
            index = self.sampler.batch_indices()
        else:
            assert index != None
        
        # sample batch
        rays_batch = self.rays[index].to(self.device, non_blocking=True)
        rgbs_batch = self.rgbs[index].to(self.device, non_blocking=True)
        radii_batch = self.radii[index].to(self.device, non_blocking=True)
        lossmult_batch = self.lossmults[index].to(self.device, non_blocking=True)

        return rays_batch, rgbs_batch, radii_batch, lossmult_batch


if __name__ == "__main__":
    torch.manual_seed(444)
    np.random.seed(444)

    train_kwargs = {
        "hold_every": 8,
        "data_dir": "/workspace/dataset/nerf_llff_data/fern",
        "split": "train",
        "batch_size": 4096,
        "n_downsamples": 4,
        "downsample": 1.0,
        "is_stack": False,
        "n_vis": -1,
        "return_radii": True,
        "patch_size": 1,
        "device": "cuda"
    }

    test_kwargs = {
        "hold_every": 8,
        "data_dir": "/workspace/dataset/nerf_llff_data/fern",
        "split": "test",
        "batch_size": 1,
        "n_downsamples": 4,
        "downsample": 1.0,
        "is_stack": True,
        "n_vis": -1,
        "return_radii": True,
        "patch_size": 1,
        "device": "cuda"
    }

    train_dataset = MultiscaleLLFFDataset(**train_kwargs)
    test_dataset = MultiscaleLLFFDataset(**test_kwargs)

    import pdb; pdb.set_trace()
