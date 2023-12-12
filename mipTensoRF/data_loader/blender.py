import os
import json
import math
import torch
from PIL import Image
from tqdm import tqdm
from .ray_utils import *
from .base import BaseDataset, get_radius, resize_and_normalize


class BlenderDataset(BaseDataset):
    def __init__(self, **kwargs):
        self.white_bg = True
        self.near_far = [2.0, 6.0]
        self.aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])

        # OpenGL (x: right, y: up, z: backward)
        # OpenCV (x: right, y: down, z: forward)
        self.blender2opencv = torch.tensor([
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ], dtype=torch.float32)

        super(BlenderDataset, self).__init__(**kwargs)
    

class SinglescaleBlenderDataset(BlenderDataset):
    def _load_renderings(self):
        assert self.n_downsamples == 1 and self.patch_size == 1

        # read dataset
        with open(os.path.join(self.data_dir, f"transforms_{self.split}.json"), "r") as f:
            meta = json.load(f)

        # image width and height after downsampling
        width, height = int(800/self.downsample), int(800/self.downsample)

        # half_width / f = tan(half_camera_angle_x)
        focal = 0.5 * width / np.tan(0.5 * meta["camera_angle_x"])

        # normalized ray directions from origin to the center of each pixel
        directions = get_ray_directions(height, width, [focal, focal])

        # radius at each pixel
        if self.return_radii:
            radii = []
            radius = get_radius(directions)
        else:
            radii = None

        # get image every img_eval_interval
        img_eval_interval = 1 if self.n_vis < 0 else len(meta["frames"]) // self.n_vis
        indices = list(range(0, len(meta["frames"]), img_eval_interval))

        # main loop
        rays, rgbs = [], []
        for i in tqdm(indices, position=0, leave=True, desc=f"{self.split:4s} data"):
            # get one frame
            frame = meta["frames"][i]

            # camera pose (c2w) in OpenCV convention
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32) @ self.blender2opencv

            # load and preprocess image
            rgb = Image.open(os.path.join(self.data_dir, f"{frame['file_path']}.png"))
            rgbs += [resize_and_normalize(rgb, width, height, self.downsample, not self.is_stack)[-1]]

            # rays in world coordinate: (H*W, 6)
            ray_o, ray_d = get_rays(directions, c2w)
            rays += [torch.cat([ray_o, ray_d], dim=1)]

            # radius: (H*W, 1)
            if self.return_radii:
                radii += [radius.view(-1, 1)]
        
        # concatenate
        if not self.is_stack:
            self.rays = torch.cat(rays, dim=0)          # (N*H*W, 3)
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
    

class MultiscaleBlenderDataset(BlenderDataset):
    def _load_renderings(self):
        # sanity check
        assert self.n_downsamples == 4 and self.downsample == 1.0
        assert self.return_radii == True

        # read dataset
        with open(os.path.join(self.data_dir, f"transforms_{self.split}.json"), "r") as f:
            meta = json.load(f)

        # width and height of the original image
        width = height = 800

        # width and height of each patch
        pW = pH = self.patch_size

        # half_width / f = tan(half_camera_angle_x)
        focal = 0.5 * width / np.tan(0.5 * meta["camera_angle_x"])

        # get image every img_eval_interval
        img_eval_interval = 1 if self.n_vis < 0 else len(meta["frames"]) // self.n_vis
        indices = list(range(0, len(meta["frames"]), img_eval_interval))

        # main loop
        rays, rgbs, radii, lossmults = [], [], [], []
        for i in tqdm(indices, position=0, leave=True, desc=f"{self.split:4s} data"):
            # get one frame
            frame = meta["frames"][i]

            # camera pose (c2w) in OpenCV convention
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32) @ self.blender2opencv

            # load a single image
            rgb = Image.open(os.path.join(self.data_dir, f"{frame['file_path']}.png"))

            # downsample to 4 different scales
            for j in range(self.n_downsamples):
                # rescale width and height
                iW, iH = width // (2**j), height // (2**j)

                # rescale, normalize, blend alpha: (ImgH, ImgW, 3) or (ImgH*ImgW, 3)
                rgb, _rgb = resize_and_normalize(rgb, iW, iH, 2**j, not self.is_stack)
                
                # rescale focal length
                fx, fy = focal / (2**j) * pW, focal / (2**j) * pH

                # ray direction from origin to the center of each pixel: (ImgH*PatchH, ImgW*PatchW, 3)
                direction = get_ray_directions(iH * pH, iW * pW, [fy, fx])

                # ray origins and directions (not normalized): (ImgH*PatchH*ImgW*PatchW, 6)
                ray_o, ray_d = get_rays(direction, c2w)
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


class VideoBlenderDataset(BlenderDataset):
    def _load_renderings(self):
        assert self.is_stack == True

        # translation mat
        trans_t = lambda t: torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

        # rotation mat
        rot_phi = lambda phi: torch.tensor([
            [1, 0, 0, 0],
            [0, math.cos(phi), -math.sin(phi), 0],
            [0, math.sin(phi), math.cos(phi), 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

        # rotation mat
        rot_theta = lambda th: torch.tensor([
            [math.cos(th), 0, -math.sin(th), 0],
            [0, 1, 0, 0],
            [math.sin(th), 0, math.cos(th), 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)
        
        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*math.pi) @ c2w
            c2w = rot_theta(theta/180.*math.pi) @ c2w
            c2w = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.float32) @ c2w
            return c2w
        
        # read camera angle
        with open(os.path.join(self.data_dir, f"transforms_{self.split}.json"), "r") as f:
            camera_angle_x = json.load(f)["camera_angle_x"]
        
        # output image resolution
        width, height = int(800/self.downsample), int(800/self.downsample)

        # focal length scaling factor
        n_frames = 30
        # focal_scale_factors = torch.cat([
        #     torch.tensor([100] * n_frames),     # coarse frame
        #     torch.linspace(100, 800, n_frames//3), # zooming in
        #     torch.tensor([800] * n_frames),     # fine frame
        #     torch.linspace(800, 100, n_frames//3)  # zooming out
        # ]).repeat(2)

        # focal_scale_factors = torch.cat([
        #     torch.tensor([400] * n_frames),
        #     torch.linspace(400, 800, n_frames//3),
        #     torch.tensor([800] * n_frames),
        #     torch.linspace(800, 100, n_frames//3),
        #     torch.tensor([100] * n_frames),
        #     torch.linspace(100, 800, n_frames//3),
        #     torch.tensor([800] * n_frames),
        #     torch.linspace(800, 400, n_frames//3)
        # ])

        focal_scale_factors = torch.cat([
            torch.tensor([200] * n_frames),
            torch.linspace(200, 800, n_frames//3),
            torch.tensor([800] * n_frames),
            torch.linspace(800, 100, n_frames//3),
            torch.tensor([100] * n_frames),
            torch.linspace(100, 800, n_frames//3),
            torch.tensor([800] * n_frames),
            torch.linspace(800, 200, n_frames//3)
        ])

        # # cropping area size
        # crop_scale_factors = torch.cat([
        #     torch.tensor([1/4.] * n_frames),
        #     torch.linspace(1/4., 1/2., n_frames//3),
        #     torch.tensor([1/2.] * n_frames),
        #     torch.linspace(1/2., 1/4., n_frames//3)
        # ]).repeat(2)

        # crop_scale_factors = torch.cat([
        #     torch.tensor([1/2.] * n_frames),
        #     torch.tensor([1/2.] * (n_frames//3)),
        #     torch.tensor([1/2.] * n_frames),
        #     torch.linspace(1/2., 1/4., n_frames//3),
        #     torch.tensor([1/4.] * n_frames),
        #     torch.linspace(1/4., 1/2., n_frames//3),
        #     torch.tensor([1/2.] * n_frames),
        #     torch.tensor([1/2.] * (n_frames//3))
        # ])

        # crop_scale_factors = torch.cat([
        #     torch.tensor([5/8.] * n_frames),
        #     torch.linspace(5/8., 1/2., n_frames//3),
        #     torch.tensor([1/2.] * n_frames),
        #     torch.linspace(1/2., 1/4., n_frames//3),
        #     torch.tensor([1/4.] * n_frames),
        #     torch.linspace(1/4., 1/2., n_frames//3),
        #     torch.tensor([1/2.] * n_frames),
        #     torch.linspace(1/2., 5/8., n_frames//3)
        # ])

        crop_scale_factors = torch.cat([
            torch.tensor([3/8.] * n_frames),
            torch.linspace(3/8., 4/8., n_frames//3),
            torch.tensor([4/8.] * n_frames),
            torch.linspace(4/8., 2/8., n_frames//3),
            torch.tensor([2/8.] * n_frames),
            torch.linspace(2/8., 4/8., n_frames//3),
            torch.tensor([4/8.] * n_frames),
            torch.linspace(4/8., 3/8., n_frames//3)
        ])

        # camera angle
        # thetas = torch.cat([
        #     torch.linspace(-180, -135, n_frames),
        #     torch.linspace(-135, -90, n_frames//3),
        #     torch.linspace(-90, -45, n_frames),
        #     torch.linspace(-45, 0, n_frames//3),
        #     torch.linspace(0, 45, n_frames),
        #     torch.linspace(45, 90, n_frames//3),
        #     torch.linspace(90, 135, n_frames),
        #     torch.linspace(135, 180, n_frames//3)
        # ])

        # thetas = torch.cat([
        #     torch.linspace(-135, -90, n_frames),
        #     torch.linspace(-90, -45, n_frames//3),
        #     torch.linspace(-45, 0, n_frames),
        #     torch.linspace(0, 45, n_frames//3),
        #     torch.linspace(45, 90, n_frames),
        #     torch.linspace(90, 135, n_frames//3),
        #     torch.linspace(135, 180, n_frames),
        #     torch.linspace(-180, -135, n_frames//3)
        # ])

        # thetas = torch.cat([
        #     torch.linspace(135, 180, n_frames),
        #     torch.linspace(-180, -135, n_frames//3),
        #     torch.linspace(-135, -90, n_frames),
        #     torch.linspace(-90, -45, n_frames//3),
        #     torch.linspace(-45, 0, n_frames),
        #     torch.linspace(0, 45, n_frames//3),
        #     torch.linspace(45, 90, n_frames),
        #     torch.linspace(90, 135, n_frames//3),
        # ])

        # thetas = torch.cat([
        #     torch.linspace(140, 160, n_frames),
        #     torch.linspace(160, 180, n_frames//3),
        #     torch.linspace(-180, -160, n_frames),
        #     torch.linspace(-160, -140, n_frames//3),
        #     torch.linspace(-140, -120, n_frames),
        #     torch.linspace(-120, 100, n_frames//3),
        #     torch.linspace(100, 120, n_frames),
        #     torch.linspace(120, 140, n_frames//3)
        # ])

        thetas = torch.cat([
            torch.linspace(140, 160, n_frames),
            torch.linspace(160, 200, n_frames//3),
            torch.linspace(-160, -140, n_frames),
            torch.linspace(-140, -120, n_frames//3),
            torch.linspace(-120, -100, n_frames),
            torch.linspace(-100, 100, n_frames//3),
            torch.linspace(100, 120, n_frames),
            torch.linspace(120, 140, n_frames//3)
        ])

        # thetas = torch.cat([
        #     torch.linspace(160, 180, n_frames),
        #     torch.linspace(180, )
        # ])

        phis = torch.cat([
            torch.tensor([-35] * n_frames),
            torch.linspace(-35, -25, n_frames//3),
            torch.tensor([-25] * n_frames),
            torch.linspace(-25, -35, n_frames//3)
        ]).repeat(2)

        # main loop
        rays, radii = [], []
        for i in tqdm(range(len(focal_scale_factors)), position=0, leave=True):
            # focal length
            focal = 0.5 * focal_scale_factors[i] / np.tan(0.5 * camera_angle_x)

            # ray direction
            directions = get_ray_directions(height, width, [focal, focal])

            # camera pose (c2w) in OpenCV convention
            c2w = pose_spherical(thetas[i], phis[i], 4.0) @ self.blender2opencv

            # rays in world coordinate: (H*W, 6)
            ray_o, ray_d = get_rays(directions, c2w)
            rays += [torch.cat([ray_o, ray_d], dim=1).view(height*width, 6)]

            # radius: (H*W, 1)
            if self.return_radii:
                radii += [get_radius(directions).view(height*width, 1)]

        self.rays = rays
        self.radii = radii
        self.crop_scale_factors = crop_scale_factors.tolist()
    
    @torch.no_grad()
    def fetch_data(self, index=None):
        ray_batch = self.rays[index].to(self.device, non_blocking=True)
        if self.return_radii == True:
            radii_batch = self.radii[index].to(self.device, non_blocking=True)
        else:
            radii_batch = None
        scale_batch = self.crop_scale_factors[index]
        return ray_batch, radii_batch, scale_batch



if __name__ == "__main__":
    torch.manual_seed(444)
    np.random.seed(444)

    # train_kwargs = {
    #     "data_dir": "/workspace/dataset/nerf_synthetic/chair",
    #     "split": "train",
    #     "batch_size": 4096,
    #     "n_downsamples": 4,
    #     "downsample": 1.0,
    #     "is_stack": False,
    #     "n_vis": -1,
    #     "return_radii": True,
    #     "patch_size": 1,
    #     "device": "cuda"
    # }

    # test_kwargs = {
    #     "data_dir": "/workspace/dataset/nerf_synthetic/chair",
    #     "split": "test",
    #     "batch_size": 1,
    #     "n_downsamples": 4,
    #     "downsample": 1.0,
    #     "is_stack": True,
    #     "n_vis": 5,
    #     "return_radii": True,
    #     "patch_size": 1,
    #     "device": "cuda"
    # }

    # train_dataset = MultiscaleBlenderDataset(**train_kwargs)
    # test_dataset = MultiscaleBlenderDataset(**test_kwargs)


    test_kwargs = {
        "data_dir": "/workspace/dataset/nerf_synthetic/chair",
        "split": "test",
        "batch_size": 1,
        "n_downsamples": 1,
        "downsample": 1.0,
        "is_stack": True,
        "n_vis": -1,
        "return_radii": True,
        "patch_size": 1,
        "device": "cuda"
    }

    video_dataset = VideoBlenderDataset(**test_kwargs)


    import pdb; pdb.set_trace()


