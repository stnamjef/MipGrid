import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x, n_freqs):
    f = (2**torch.arange(n_freqs, dtype=torch.float32, device=x.device))
    fx = torch.flatten(f[None, :, None] * x[:, None, :], 1, -1)
    return torch.cat([torch.sin(fx), torch.cos(fx)], dim=-1)


def normalize_coords(pts, low, up):
    return (pts - low) / (up - low) * 2 - 1


def cumprod_exclusive(tensor):
    # https://github.com/krrish94/nerf-pytorch
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0
    return cumprod


class AlphaMask(nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaMask, self).__init__()
        self.aabb = aabb
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:]).to(device)
        self.inv_grid_size = torch.tensor(alpha_volume.shape, dtype=torch.int64).to(device)

    def forward(self, pts):
        pts = normalize_coords(pts, *self.aabb).view(1, -1, 1, 1, 3)
        alpha = F.grid_sample(self.alpha_volume, pts, align_corners=True).view(-1)
        return alpha
    

class ToRGB(nn.Module):
    def __init__(self, in_features, feat_n_freqs, view_n_freqs, hidden_features):
        super(ToRGB, self).__init__()

        self.in_features = 2 * feat_n_freqs * in_features + \
                           2 * view_n_freqs * 3 + \
                           in_features + 3
        
        self.feat_n_freqs = feat_n_freqs
        self.view_n_freqs = view_n_freqs
        
        self.net = nn.Sequential(
            nn.Linear(self.in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 3),
            nn.Sigmoid()
        )

    def forward(self, features, viewdirs):
        x = [features, viewdirs]
        if self.feat_n_freqs > 0:
            x += [positional_encoding(features, self.feat_n_freqs)]
        if self.view_n_freqs > 0:
            x += [positional_encoding(viewdirs, self.view_n_freqs)]
        x = torch.cat(x, dim=-1)
        return self.net(x)


class TensorBase(nn.Module):
    def __init__(
        self,
        device,
        aabb,
        white_bg,
        near_far,
        step_ratio,
        grid_size,
        den_channels,
        app_channels,
        density_shift,
        density_activation,
        feat_n_freqs,
        view_n_freqs,
        to_rgb_in_features,
        to_rgb_hidden_features,
        distance_scale,
        alpha_mask_threshold,
        raymarch_weight_threshold,
        alpha_mask=None,
        t_min=None,
        t_max=None,
        n_samples=None,
    ):
        super(TensorBase, self).__init__()

        self.t_min = t_min
        self.t_max = t_max
        self.n_samples = n_samples

        self.device = device
        self.aabb = aabb.to(device)
        self.white_bg = white_bg
        self.near_far = near_far
        self.step_ratio = step_ratio
        self.update_step_size(grid_size)
        
        self.den_channels = den_channels
        self.app_channels = app_channels

        self.density_shift = density_shift
        self.density_activation = density_activation

        self.feat_n_freqs = feat_n_freqs
        self.view_n_freqs = view_n_freqs
        self.to_rgb_in_features = to_rgb_in_features
        self.to_rgb_hidden_features = to_rgb_hidden_features

        self.distance_scale = distance_scale
        self.alpha_mask_threshold = alpha_mask_threshold
        self.raymarch_weight_threshold = raymarch_weight_threshold

        self.mat_mode = [[0, 1], [0, 2], [1, 2]]
        self.vec_mode = [2, 1, 0]

        self.to_rgb = ToRGB(
            to_rgb_in_features, feat_n_freqs,
            view_n_freqs, to_rgb_hidden_features
        ).to(device)

        self.alpha_mask = alpha_mask

    def get_kwargs(self):
        kwargs = {
            "aabb": self.aabb,
            "white_bg": self.white_bg,
            "near_far": self.near_far,
            "step_ratio": self.step_ratio,
            "grid_size": self.grid_size,
            # grid
            "den_channels": self.den_channels,
            "app_channels": self.app_channels,
            # density rendering
            "density_shift": self.density_shift,
            "density_activation": self.density_activation,
            # appearance rendering
            "feat_n_freqs": self.feat_n_freqs,
            "view_n_freqs": self.view_n_freqs,
            "to_rgb_in_features": self.to_rgb_in_features,
            "to_rgb_hidden_features": self.to_rgb_hidden_features,
            # rendering misc
            "distance_scale": self.distance_scale,
            "alpha_mask_threshold": self.alpha_mask_threshold,
            "raymarch_weight_threshold": self.raymarch_weight_threshold,
            # minmax and number of samples
            "t_min": self.t_min,
            "t_max": self.t_max,
            "n_samples": self.n_samples
        }
        return kwargs
    
    def get_params(self, lr_init_grid=0.02, lr_init_network=0.001):
        raise NotImplementedError
    
    def save(self, path):
        ckpt = {"kwargs": self.get_kwargs(), "state_dict": self.state_dict()}
        if self.alpha_mask is not None:
            alpha_volume = self.alpha_mask.alpha_volume.bool().cpu().numpy()
            ckpt.update({"alpha_mask.shape": alpha_volume.shape})
            ckpt.update({"alpha_mask.mask": np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({"alpha_mask.aabb": self.alpha_mask.aabb.cpu()})
        torch.save(ckpt, path)
    
    def load(self, ckpt):
        if "alpha_mask.aabb" in ckpt.keys():
            length = np.prod(ckpt["alpha_mask.shape"])
            alpha_volume = torch.from_numpy(np.reshape(
                np.unpackbits(ckpt["alpha_mask.mask"])[:length],
                newshape=ckpt["alpha_mask.shape"]
            ))
            self.alpha_mask = AlphaMask(
                self.device, ckpt["alpha_mask.aabb"],
                alpha_volume.float()
            )
        self.load_state_dict(ckpt["state_dict"])
    
    def update_step_size(self, grid_size):
        # size of a volume in (X, Y, Z) order
        self.grid_size = grid_size.to(self.device)

        # length of aabb per each cell
        self.units = (self.aabb[1] - self.aabb[0])/ (self.grid_size - 1)

        # determine space between ts, "near + arange(n_samples) * step_size" is equal to
        # "near * (1 - t) + far * t", where t = linspace(0, 1 * step_size, n_samples)
        self.step_size = torch.mean(self.units) * self.step_ratio

        # logging
        print(f"=====> UPDATING STEP SIZE ...")
        print(f"  aabb: {self.aabb.view(-1)}")
        print(f"  grid size: {self.grid_size}")
        print(f"  sampling step size: {self.step_size}")

    def update_t_distance_radius(self, rays, radii=None):
        assert rays.ndim == 2

        # unpack
        near, far = self.near_far

        # split ray into origin and direction
        ray_o, ray_d = rays[:, :3], rays[:, 3:]

        # replace 0s with a very small value to prevent divide by zero
        d = torch.where(ray_d == 0, torch.tensor(1e-6), ray_d)

        # t = (o + td - o) / d: (2, n_rays, 3)
        rates = (self.aabb[..., None, :].cpu() - ray_o) / d

        # min along dim=0 & max along dim=-1: (n_rays,)
        t_mins = rates.amin(0).amax(-1).clamp(near, far)
        t_maxs = rates.amax(0).amin(-1).clamp(near, far)

        # set class attribute
        self.t_min = float(torch.min(t_mins))
        self.t_max = float(torch.max(t_maxs))
    
    @torch.no_grad()
    def update_alpha_mask(self, grid_size):
        # int64 tensor to Size
        grid_size = torch.Size(grid_size)
        
        # dense points inside a volume
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, grid_size[0], device=self.device),
            torch.linspace(0, 1, grid_size[1], device=self.device),
            torch.linspace(0, 1, grid_size[2], device=self.device),
            indexing="ij"
        ), dim=-1)

        # move points into aabb
        pts = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        # output tensor
        alpha = torch.zeros(grid_size, device=self.device)

        # compute alpha values
        for i in range(grid_size[0]):
            alpha[i] = self._calc_alpha(pts[i].view(-1, 3)).view(grid_size[1:])
        
        # clamp and reshape: (W, H, D) --> (1, 1, W, H, D)
        alpha = alpha.clamp(0, 1)[None, None]

        # max-pool alpha volume
        alpha = F.max_pool3d(alpha, 3, 1, 1).view(grid_size)

        # thresholding (binarization)
        alpha = torch.where(alpha >= self.alpha_mask_threshold, 1., 0.)

        # update, NOTE: (W, H, D) --> (D, H, W) for grid_sample
        self.alpha_mask = AlphaMask(self.device, self.aabb, alpha.transpose(0, 2).contiguous())

        # get valid points, where alpha == 1
        valid_pts = pts[alpha == 1]

        # update axis-aligned bounding box: (2, 3)
        new_aabb = torch.stack([valid_pts.amin(0), valid_pts.amax(0)])

        # save how much pts are left
        total_n_voxels = grid_size[0] * grid_size[1] * grid_size[2]
        alpha_rest = torch.sum(alpha) / total_n_voxels * 100

        # logging
        print(f"  bbox: {new_aabb.view(-1)}, alpha rest {alpha_rest:.2f}%")

        return new_aabb
    
    @torch.no_grad()
    def _calc_alpha(self, pts):
        if self.alpha_mask is not None:
            valid_pts = self.alpha_mask(pts) > 0
        else:
            valid_pts = torch.ones(pts.shape[:-1], dtype=bool, device=self.device)
        
        sigma = torch.zeros(pts.shape[:-1], device=self.device)

        if torch.any(valid_pts):
            def act_fn(features):
                if self.density_activation == "softplus":
                    sigma = F.softplus(features + self.density_shift)
                elif self.density_activation == "relu":
                    sigma = F.relu(features)
                else:
                    raise ValueError("Invalid density_activation")
                return sigma
            pts = normalize_coords(pts, *self.aabb)
            sigma[valid_pts] = act_fn(self.sample_den_features(pts[valid_pts], None))
        
        alpha = 1 - torch.exp(-sigma * self.step_size).view(pts.shape[:-1])

        return alpha
    
    @torch.no_grad()
    def filter_rays(self, rays, rgbs, *aux_data, n_samples=256, batch=10240*5, bbox_only=False):
        assert rays.ndim == 2

        print("=====> FILTERING RAYS ...")
        begin = time.time()

        in_bbox = []
        for i in torch.split(torch.arange(len(rays)), batch):
            # get a batch of rays and ship to device
            rays_batch = rays[i].to(self.device)

            # split into origin and direction
            r_o, r_d = rays_batch[..., :3], rays_batch[..., 3:]

            if bbox_only:  # filter with aabb
                # replace 0s with a very small value to prevent divide by zero
                d = torch.where(r_d == 0, torch.tensor(1e-6, device=self.device), r_d)

                # t = (o + td - o) / d: (2, n_rays, 3)
                rate = (self.aabb[..., None, :] - r_o) / d

                # get min & max values
                t_min = rate.amin(0).amax(-1)
                t_max = rate.amax(0).amin(-1)

                # rays inside bbox
                mask = t_max > t_min
            else:  # filter with alpha mask (never use when rays are defined in ndc space)
                pts, _, _ = self._sample_along_rays(r_o, r_d, len(r_o), n_samples, False, False)
                mask = (self.alpha_mask(pts).view(pts.shape[:-1]) > 0).any(-1)

            # save mask
            in_bbox.append(mask.cpu())
        
        # concatenate all masks
        in_bbox = torch.cat(in_bbox)

        # filter rays & others
        rays = rays[in_bbox]
        rgbs = rgbs[in_bbox]

        # radii and loss_mults
        aux_data = (data[in_bbox] if data is not None else None for data in aux_data)

        # logging
        time_elapsed = time.time() - begin
        mask_ratio = torch.sum(in_bbox) / len(rays) * 100
        print(f"  Ray filtering done! takes {time_elapsed:.4f} sec., mask ratio {mask_ratio:2.2f}%")

        return rays, rgbs, *aux_data
    
    def sample_den_features(self, pts, scales=None):
        raise NotImplementedError
    
    def sample_app_features(self, pts, scales=None):
        raise NotImplementedError
    
    def _sample_along_rays(self, ray_o, ray_d, n_rays, n_samples, perturb, ndc_ray, lindisp=False):
        # t values in p = o + td
        t = torch.linspace(0, 1, n_samples, device=self.device)

        if ndc_ray:  # whole scene
            near, far = self.near_far
        else:  # axis-aligned bbox
            near, far = self.t_min, self.t_max

        if lindisp:  # linear in disparity (inverse depth)
            t = 1. / (1. / near * (1. - t) + 1. / far * t)
        else:  # linear in depth
            t = near * (1. - t) + far * t

        # broadcast: (n_samples,) --> (n_rays, n_samples)
        t = torch.broadcast_to(t, (n_rays, n_samples))
        
        if perturb:  # stratified sampling
            t_mid = 0.5 * (t[..., 1:] + t[..., :-1])
            upper = torch.cat([t_mid, t[..., -1:]], dim=-1)
            lower = torch.cat([t[..., :1], t_mid], dim=-1)
            t_rand = torch.rand((n_rays, n_samples), device=self.device)
            t_vals = lower + (upper - lower) * t_rand
        else:
            t_vals = t

        # points in world coordinate (p = o + td)
        pts = ray_o[..., None, :] + t_vals[..., :, None] * ray_d[..., None, :]

        # a mask indicates if points are inside the bounding box: (n_rays, n_samples)
        in_bbox = torch.all((self.aabb[0] <= pts) & (pts <= self.aabb[1]), dim=-1)

        # points inside bbox "and" alpha == 1
        if self.alpha_mask is not None:
            # in_bbox[in_bbox] &= (self.alpha_mask(pts[in_bbox]) > 0)
            alpha_mask = self.alpha_mask(pts[in_bbox]) > 0
            out_bbox = ~in_bbox
            out_bbox[in_bbox] |= (~alpha_mask)
            in_bbox = ~out_bbox

        return pts, t_vals, in_bbox

    def _render_rays(self, n_rays, n_samples, in_bbox, pts, dist, viewdirs, is_train, scales=None):
        def act_fn(features):
            if self.density_activation == "softplus":
                sigma = F.softplus(features + self.density_shift)
            elif self.density_activation == "relu":
                sigma = F.relu(features)
            else:
                raise ValueError("Invalid density_activation")
            return sigma
        
        # density tensor
        sigma = torch.zeros((n_rays, n_samples), device=self.device)

        if torch.any(in_bbox):
            if scales is None:
                sigma[in_bbox] = act_fn(self.sample_den_features(pts[in_bbox]))
            else:
                # masking
                scales_masked = [s[in_bbox] for s in scales]
                sigma[in_bbox] = act_fn(self.sample_den_features(pts[in_bbox], scales_masked))

        # opacity of samples along each ray
        alpha = 1. - torch.exp(-sigma * dist * self.distance_scale)

        # transmittance * opacity: T * (1 - exp(-sigma*dist))
        weight = alpha * cumprod_exclusive(1. - alpha + 1e-10)

        # a mask indicates at which point to end raymarching
        app_mask = weight > self.raymarch_weight_threshold

        # appearance tensor
        rgb = torch.zeros((n_rays, n_samples, 3), device=self.device)
        
        if torch.any(app_mask):
            # expand viewdir: (n_rays, 3) --> (n_rays, n_samples, 3)
            viewdirs = viewdirs[:, None, :].expand(-1, n_samples, -1)

            if scales is None:
                rgb[app_mask] = self.to_rgb(self.sample_app_features(pts[app_mask]), viewdirs[app_mask])
            else:
                # masking
                scales_masked = [s[app_mask] for s in scales]
                rgb[app_mask] = self.to_rgb(self.sample_app_features(pts[app_mask], scales_masked), viewdirs[app_mask])
        
        # sum of weights along each ray
        acc_map = torch.sum(weight, dim=-1)
        rgb_map = torch.sum(weight[..., None] * rgb, dim=-2)

        # to composite onto a white background, use the acc alpha map
        if self.white_bg or (is_train and torch.rand(1) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = torch.clamp(rgb_map, 0, 1)

        return acc_map, rgb_map, weight
    
    def forward(self, ray_batch, n_samples, is_train, ndc_ray, radii=None):
        # assert radii is None

        # jittered ray
        is_jittered = False
        if ray_batch.ndim == 3:
            is_jittered = True
            patch_hw = ray_batch.shape[-2]
            ray_batch = ray_batch.view(-1, 6)
        
        # for test
        if n_samples < 0:
            n_samples = self.n_samples

        # split rays into origin and direction
        ray_o, ray_d = ray_batch[:, 0:3], ray_batch[:, 3:6]

        # sample points along each ray and get mask
        pts, t_vals, in_bbox = self._sample_along_rays(ray_o, ray_d, len(ray_batch), n_samples, is_train, ndc_ray)

        # distance between sample points: (n_rays, n_samples)
        dist = torch.diff(t_vals, dim=-1, append=t_vals[:, -1:])

        # norm of ray direction
        ray_norm = torch.norm(ray_d, dim=-1, keepdim=True)

        # convert to real-world dist & normalize ray_d
        dist, ray_d = dist * ray_norm, ray_d / ray_norm

        # normalize coordinates
        pts = normalize_coords(pts, *self.aabb)

        # render rays
        acc_map, rgb_map, weight = self._render_rays(len(ray_batch), n_samples, in_bbox, pts, dist, ray_d, is_train)

        # depth map
        with torch.no_grad():
            depth_map = torch.sum(weight * t_vals, dim=-1)
            depth_map = depth_map + (1. - acc_map) * ray_batch[..., -1]  # ??

        # jittered ray
        if is_jittered:
            rgb_map = rgb_map.view(-1, patch_hw, 3).mean(1)
            depth_map = depth_map.view(-1, patch_hw).mean(1)
        
        return rgb_map, depth_map


class MipTensorBase(TensorBase):
    def __init__(
        self,
        n_kernels,
        kernel_size,
        kernel_init,
        apply_kernel,
        learnable_kernel,
        scale_types,
        d_min=None,
        d_max=None,
        cylinder_r_min=None,
        cylinder_r_max=None,
        cone_r_min=None,
        cone_r_max=None,
        **kwargs
    ):
        super(MipTensorBase, self).__init__(**kwargs)
        
        self.d_min = d_min
        self.d_max = d_max
        self.cylinder_r_min = cylinder_r_min
        self.cylinder_r_max = cylinder_r_max
        self.cone_r_min = cone_r_min
        self.cone_r_max = cone_r_max

        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.kernel_init = kernel_init
        self.apply_kernel = apply_kernel
        self.learnable_kernel = learnable_kernel
        self.scale_types = scale_types

    def get_kwargs(self):
        kwargs = super(MipTensorBase, self).get_kwargs()
        kwargs.update({
            "d_min": self.d_min,
            "d_max": self.d_max,
            "cylinder_r_min": self.cylinder_r_min,
            "cylinder_r_max": self.cylinder_r_max,
            "cone_r_min": self.cone_r_min,
            "cone_r_max": self.cone_r_max,

            "n_kernels": self.n_kernels,
            "kernel_size": self.kernel_size,
            "kernel_init": self.kernel_init,
            "apply_kernel": self.apply_kernel,
            "learnable_kernel": self.learnable_kernel,
            "scale_types": self.scale_types
        })
        return kwargs
    
    def update_t_distance_radius(self, rays, radii=None):
        assert rays.ndim == 2 and radii != None

        # unpack
        near, far = self.near_far

        # split ray into origin and direction
        ray_o, ray_d = rays[:, :3], rays[:, 3:]

        # replace 0s with a very small value to prevent divide by zero
        d = torch.where(ray_d == 0, torch.tensor(1e-6), ray_d)

        # t = (o + td - o) / d: (2, n_rays, 3)
        rates = (self.aabb[..., None, :].cpu() - ray_o) / d

        # min along dim=0 & max along dim=-1: (n_rays,)
        t_mins = rates.amin(0).amax(-1).clamp(near, far)
        t_maxs = rates.amax(0).amin(-1).clamp(near, far)

        # set class attribute
        self.t_min = float(torch.min(t_mins))
        self.t_max = float(torch.max(t_maxs))

        # min & max of distance from origin to sampled points: norm((o + td) - o)
        d_mins = torch.norm(t_mins[:, None] * ray_d, dim=-1, keepdim=True)
        d_maxs = torch.norm(t_maxs[:, None] * ray_d, dim=-1, keepdim=True)

        # set class attribute
        self.d_min = float(torch.min(d_mins))
        self.d_max = float(torch.max(d_maxs))

        self.cylinder_r_min = float(torch.min(radii))
        self.cylinder_r_max = float(torch.max(radii))

        # min & max of cone radius: radii * norm(td)
        self.cone_r_min = float(torch.min(radii * d_mins))
        self.cone_r_max = float(torch.max(radii * d_maxs))

    def forward(self, ray_batch, n_samples, is_train, ndc_ray, radii=None):
        assert radii is not None

        # jittered ray
        is_jittered = False
        if ray_batch.ndim == 3:
            is_jittered = True
            patch_hw = ray_batch.shape[-2]
            ray_batch = ray_batch.view(-1, 6)

        # for test
        if n_samples < 0:
            n_samples = self.n_samples

        # split rays into origin and direction
        ray_o, ray_d = ray_batch[:, 0:3], ray_batch[:, 3:6]

        # sample points along each ray and get mask
        pts, t_vals, in_bbox = self._sample_along_rays(ray_o, ray_d, len(ray_batch), n_samples, is_train, ndc_ray)

        # distance between sample points: (n_rays, n_samples)
        dist = torch.diff(t_vals, dim=-1, append=t_vals[:, -1:])

        # norm of ray direction
        ray_norm = torch.norm(ray_d, dim=-1, keepdim=True)

        # convert to real-world dist & normalize ray_d
        dist, ray_d = dist * ray_norm, ray_d / ray_norm

        # normalize coordinates
        pts = normalize_coords(pts, *self.aabb)

        if self.scale_types is not None and self.apply_kernel is not None:
            scales = []  # prepare scales
            if "distance" in self.scale_types:
                # origin to points: norm((o + td) - o)
                # (n_rays, n_samples, 1) * (n_rays, 1, 1)
                dist_to_origin = t_vals[..., :, None] * ray_norm[..., :, None]
                scales.append(normalize_coords(dist_to_origin, self.d_min, self.d_max))
            
            if "cylinder_radius" in self.scale_types:
                # cylinder radius: (n_rays, 1) --> (n_rays, n_samples, 1)
                cylinder_r = radii[:, None, :].expand(-1, n_samples, -1)
                scales.append(normalize_coords(cylinder_r, self.cylinder_r_min, self.cylinder_r_max))
            
            if "cone_radius" in self.scale_types:
                # cone radius: radii * norm(td)
                # (n_rays, 1, 1) * (n_rays, n_samples, 1) * (n_rays, 1, 1)
                cone_r = radii[..., :, None] * t_vals[..., :, None] * ray_norm[..., :, None]
                scales.append(normalize_coords(cone_r, self.cone_r_min, self.cone_r_max))
        else:
            scales = None

        # render rays
        acc_map, rgb_map, weight = self._render_rays(len(ray_batch), n_samples, in_bbox, pts, dist, ray_d, is_train, scales)

        # depth map
        with torch.no_grad():
            depth_map = torch.sum(weight * t_vals, dim=-1)
            depth_map = depth_map + (1. - acc_map) * ray_batch[..., -1]  # ??

        # jittered ray
        if is_jittered:
            rgb_map = rgb_map.view(-1, patch_hw, 3).mean(1)
            depth_map = depth_map.view(-1, patch_hw).mean(1)
        
        return rgb_map, depth_map
