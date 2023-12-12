import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tensor_base import TensorBase, MipTensorBase


class TensorVM(TensorBase):
    def __init__(self, **kwargs):
        super(TensorVM, self).__init__(**kwargs)

        self.den_mats, self.den_vecs = self._init_VM(self.den_channels, 0.1)
        self.app_mats, self.app_vecs = self._init_VM(self.app_channels, 0.1)
        self.basis_mat = nn.Linear(sum(self.app_channels), self.to_rgb_in_features, bias=False, device=self.device)
    
    def _init_VM(self, channels, scale):
        assert len(channels) == 3

        mats, vecs = [], []
        for i, ch in enumerate(channels):
            v_size = self.grid_size[self.vec_mode[i]]
            m_size0, m_size1 = self.grid_size[self.mat_mode[i]]
            mats.append(nn.Parameter(scale * torch.randn((1, ch, m_size1, m_size0))))
            vecs.append(nn.Parameter(scale * torch.randn((1, ch, v_size, 1))))
        
        return nn.ParameterList(mats).to(self.device), nn.ParameterList(vecs).to(self.device)

    def get_params(self, lr_init_grid=0.02, lr_init_network=0.001):
        params = [
            {"params": self.den_mats, "lr": lr_init_grid},
            {"params": self.den_vecs, "lr": lr_init_grid},
            {"params": self.app_mats, "lr": lr_init_grid},
            {"params": self.app_vecs, "lr": lr_init_grid},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
            {"params": self.to_rgb.parameters(), "lr": lr_init_network}
        ]
        return params
    
    def _setup_coordinates(self, n_samples, pts):
        # xy, xz, yz: (n_samples, 1, 3, 2) --> (3, n_samples, 1, 2)
        coord_mat = pts[:, None, self.mat_mode].permute(2, 0, 1, 3)

        # 0z, 0y, 0x: (3, n_samples, 1, 2)
        zeros = torch.zeros((n_samples, 3, 1), device=self.device)
        coord_vec = torch.stack([zeros, pts[:, self.vec_mode, None]], dim=-1).permute(1, 0, 2, 3)

        return coord_mat, coord_vec
    
    def sample_den_features(self, pts, scales=None):
        # number of points sampled
        n_samples = len(pts)

        # sampling positions: (3, n_samples, 1, 2)
        coord_mat, coord_vec = self._setup_coordinates(n_samples, pts)

        # output tensor
        den_features = torch.zeros((n_samples,), device=self.device)

        # sample density features
        for i in range(3):
            mat = F.grid_sample(self.den_mats[i], coord_mat[[i]], align_corners=True).view(-1, n_samples)
            vec = F.grid_sample(self.den_vecs[i], coord_vec[[i]], align_corners=True).view(-1, n_samples)
            den_features = den_features + torch.sum(mat * vec, dim=0)

        return den_features
    
    def sample_app_features(self, pts, scales=None):
        # number of points sampled
        n_samples = len(pts)

        # sampling positions: (3, n_samples, 1, 2)
        coord_mat, coord_vec = self._setup_coordinates(n_samples, pts)

        # sample appearance features
        app_features = []
        for i in range(3):
            mat = F.grid_sample(self.app_mats[i], coord_mat[[i]], align_corners=True).view(-1, n_samples)
            vec = F.grid_sample(self.app_vecs[i], coord_vec[[i]], align_corners=True).view(-1, n_samples)
            app_features.append(mat * vec)
        
        return self.basis_mat(torch.cat(app_features).T)

    @torch.no_grad()
    def upsample_volume_grid(self, grid_size):
        print("=====> UPSAMPLING FEATURE GRID ...")
        def upsample_VM(mats, vecs):
            for i in range(3):
                v0 = self.vec_mode[i]
                m0, m1 = self.mat_mode[i]
                mats[i] = nn.Parameter(F.interpolate(
                    mats[i].data, size=(grid_size[m1], grid_size[m0]),
                    mode="bilinear", align_corners=True
                ))
                vecs[i] = nn.Parameter(F.interpolate(
                    vecs[i].data, size=(grid_size[v0], 1),
                    mode="bilinear", align_corners=True
                ))
            return mats, vecs
        
        self.den_mats, self.den_vecs = upsample_VM(self.den_mats, self.den_vecs)
        self.app_mats, self.app_vecs = upsample_VM(self.app_mats, self.app_vecs)
        self.update_step_size(grid_size)
    
    @torch.no_grad()
    def shrink_volume_grid(self, new_aabb):
        print("=====> SHRINKING FEATURE GRID ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(t_l).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)

        def shrink_VM(mats, vecs):
            for i in range(3):
                v = self.vec_mode[i]
                m0, m1 = self.mat_mode[i]
                mats[i] = nn.Parameter(mats[i].data[..., t_l[m1]:b_r[m1], t_l[m0]:b_r[m0]])
                vecs[i] = nn.Parameter(vecs[i].data[..., t_l[v]:b_r[v], :])
            return mats, vecs

        self.den_mats, self.den_vecs = shrink_VM(self.den_mats, self.den_vecs)
        self.app_mats, self.app_vecs = shrink_VM(self.app_mats, self.app_vecs)
        
        # if not torch.all(self.alpha_mask.grid_size == self.grid_size):
        #     t_l_r, b_r_r = t_l / (self.grid_size - 1), (b_r - 1) / (self.grid_size - 1)
        #     correct_aabb = torch.zeros_like(new_aabb)
        #     correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
        #     correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
        #     print("  aabb", new_aabb, "\ncorrect aabb", correct_aabb)
        #     new_aabb = correct_aabb

        new_size = b_r - t_l
        self.aabb = new_aabb
        self.update_step_size(new_size)


class MipTensorVM(MipTensorBase):
    def __init__(self, **kwargs):
        super(MipTensorVM, self).__init__(**kwargs)

        self.den_mats, self.den_vecs = self._init_VM(self.den_channels, 0.1)
        self.app_mats, self.app_vecs = self._init_VM(self.app_channels, 0.1)
        self.basis_mat = nn.Linear(sum(self.app_channels), self.to_rgb_in_features, bias=False, device=self.device)
        if self.apply_kernel:
            self.init_kernels()

    def _init_VM(self, channels, scale):
        assert len(channels) == 3

        mats, vecs = [], []
        for i, ch in enumerate(channels):
            v_size = self.grid_size[self.vec_mode[i]]
            m_size0, m_size1 = self.grid_size[self.mat_mode[i]]
            mats.append(nn.Parameter(scale * torch.randn((1, ch, m_size1, m_size0))))
            vecs.append(nn.Parameter(scale * torch.randn((1, ch, v_size, 1))))
        
        return nn.ParameterList(mats).to(self.device), nn.ParameterList(vecs).to(self.device)
    
    def _init_kernel(self, channels, sigma=None, scale=None):
        def gaussian(out_channels, kernel_sizes):
            assert sigma is not None
            # The Gaussian kernel is the product of the Gaussian function of each dimension
            mgrid = torch.meshgrid([
                torch.arange(size, dtype=torch.float32) \
                    for size in kernel_sizes
            ], indexing="ij")

            kernel = 1
            for size, grid in zip(kernel_sizes, mgrid):
                mean = (size - 1) / 2
                kernel *= 1 / (sigma * math.sqrt(2 * math.pi)) * \
                    torch.exp(-0.5 * ((grid - mean) / sigma)**2)
            
            # to make sum of kernel equal to 1
            kernel = kernel / torch.sum(kernel)

            return kernel.repeat(out_channels, 1, 1, 1)
        
        def random_normal(out_channels, kernel_sizes):
            assert scale is not None
            return scale * torch.randn((out_channels, 1, *kernel_sizes))
        
        def identity(out_channels, kernel_sizes):
            height, width = kernel_sizes
            assert height % 2 != 0 and width % 2 != 0
            kernel = torch.zeros(out_channels, 1, *kernel_sizes)
            kernel[..., height//2, width//2] = 1
            return kernel
        
        if self.kernel_init == "gaussian":
            f = gaussian
        elif self.kernel_init == "random_normal":
            f = random_normal
        elif self.kernel_init == "identity":
            f = identity
        else:
            raise ValueError("Invalid init_type")
        
        mat_kernels, vec_kernels = [], []
        for ch in channels:
            mat_kernel = f(ch*self.n_kernels, (self.kernel_size, self.kernel_size)).to(self.device)
            vec_kernel = f(ch*self.n_kernels, (self.kernel_size, 1)).to(self.device)
            if self.learnable_kernel:
                mat_kernel = nn.Parameter(mat_kernel)
                vec_kernel = nn.Parameter(vec_kernel)
            mat_kernels.append(mat_kernel)
            vec_kernels.append(vec_kernel)
        
        if self.learnable_kernel:
            mat_kernels = nn.ParameterList(mat_kernels)
            vec_kernels = nn.ParameterList(vec_kernels)
        
        return mat_kernels, vec_kernels
    
    def init_kernels(self):
        assert self.scale_types is not None
        self.apply_kernel = True
        self.den_mat_kernels, self.den_vec_kernels = self._init_kernel(self.den_channels, sigma=1, scale=0.1)
        self.app_mat_kernels, self.app_vec_kernels = self._init_kernel(self.app_channels, sigma=1, scale=0.1)
    
    def get_params(self, lr_init_grid=0.02, lr_init_network=0.001):
        params = [
            {"params": self.den_mats, "lr": lr_init_grid},
            {"params": self.den_vecs, "lr": lr_init_grid},
            {"params": self.app_mats, "lr": lr_init_grid},
            {"params": self.app_vecs, "lr": lr_init_grid},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
            {"params": self.to_rgb.parameters(), "lr": lr_init_network},
        ]
        if self.apply_kernel:
            params += [
                {"params": self.den_mat_kernels, "lr": lr_init_network},
                {"params": self.den_vec_kernels, "lr": lr_init_network},
                {"params": self.app_mat_kernels, "lr": lr_init_network},
                {"params": self.app_vec_kernels, "lr": lr_init_network},
            ]
        return params
    
    def _setup_coordinates(self, n_samples, pts, scales=None):
        if scales is None:
            # xy, xz, yz coordinates: (3, n_samples, 1, 2)
            coord_mat = pts[:, None, self.mat_mode].permute(2, 0, 1, 3)

            # 0z, 0y, 0x: (3, n_samples, 1, 2)
            zeros = torch.zeros((n_samples, 3, 1), device=self.device)
            coord_vec = torch.stack([zeros, pts[:, self.vec_mode, None]], dim=-1).permute(1, 0, 2, 3)
        else:
            # xys, xzs, yzs: (n_samples, 3, 3)
            coord_mat = torch.cat([pts[:, self.mat_mode], scales[..., None].expand(-1, 3, -1)], dim=-1)

            # (n_samples, 3, 3) --> (3, n_samples, 1, 1, 3)
            coord_mat = coord_mat.permute(1, 0, 2).view(3, -1, 1, 1, 3)

            # 0zs, 0ys, 0xs: (n_samples, 3, 3)
            zeros = torch.zeros((n_samples, 3), device=self.device)
            coord_vec = torch.stack([zeros, pts[:, self.vec_mode], scales.expand(-1, 3)], dim=-1)

            # (n_samples, 3, 3) --> (3, n_samples, 1, 1, 3)
            coord_vec = coord_vec.permute(1, 0, 2).view(3, -1, 1, 1, 3)
        return coord_mat, coord_vec

    def _convolve_with_kernels(self, mats, mat_kernels, vecs, vec_kernels, channels, n_scales):
        assert self.n_kernels % n_scales == 0

        out_mats, out_vecs = [], []
        for m, mk, v, vk, ch in zip(mats, mat_kernels, vecs, vec_kernels, channels):
            # (1, ch, h, w) --> (1, ch*n_kernels, h, w)
            m = m.repeat(1, self.n_kernels, 1, 1)
            v = v.repeat(1, self.n_kernels, 1, 1)
            # apply kernel: (1, n_kernels, ch, h, w) --> (1, n_scales, ch, n_kernels//n_scales, h, w)
            _m = F.conv2d(m, mk, padding="same", groups=self.n_kernels*ch) \
                  .reshape(1, n_scales, -1, ch, *m.shape[-2:]).transpose(2, 3)
            _v = F.conv2d(v, vk, padding="same", groups=self.n_kernels*ch) \
                  .reshape(1, n_scales, -1, ch, *v.shape[-2:]).transpose(2, 3)
            out_mats.append(_m)
            out_vecs.append(_v)
        
        return out_mats, out_vecs
    
    def sample_den_features(self, pts, scales=None):
        # number of points
        n_samples = len(pts)

        if self.apply_kernel is True and scales is not None:
            # number of scales
            n_scales = len(scales)

            coord_mats = []
            coord_vecs = []
            for scale in scales:
                # sampling positions with sales: (3, n_samples, 1, 2)
                coord_mat, coord_vec = self._setup_coordinates(n_samples, pts, scale)
                coord_mats.append(coord_mat)
                coord_vecs.append(coord_vec)

            # apply kernel: 3 x (1, n_scales, ch, n_kernels//n_scales, h, w)
            mats, vecs = self._convolve_with_kernels(
                self.den_mats, self.den_mat_kernels,
                self.den_vecs, self.den_vec_kernels,
                self.den_channels, n_scales
            )
        else:
            # sampling positions without scales: (3, n_samples, 1, 2)
            coord_mats, coord_vecs = self._setup_coordinates(n_samples, pts)

            # 3 x (1, ch, h, w)
            mats, vecs = self.den_mats, self.den_vecs

        # output tensor
        den_features = torch.zeros((n_samples,), device=self.device)

        # grid sample
        for i in range(3):
            if self.apply_kernel is True and scales is not None:
                for j in range(n_scales):
                    mat = F.grid_sample(mats[i][:, j], coord_mats[j][[i]], align_corners=True).view(-1, n_samples)
                    vec = F.grid_sample(vecs[i][:, j], coord_vecs[j][[i]], align_corners=True).view(-1, n_samples)
                    den_features = den_features + torch.sum(mat * vec, dim=0)
                den_features = den_features / n_scales
            else:
                mat = F.grid_sample(mats[i], coord_mats[[i]], align_corners=True).view(-1, n_samples)
                vec = F.grid_sample(vecs[i], coord_vecs[[i]], align_corners=True).view(-1, n_samples)
                den_features = den_features + torch.sum(mat * vec, dim=0)
        
        return den_features
    
    def sample_app_features(self, pts, scales=None):
        # number of points
        n_samples = len(pts)

        if self.apply_kernel is True and scales is not None:
            # number of scales
            n_scales = len(scales)

            # sampling positions
            coord_mats = []
            coord_vecs = []
            for scale in scales:
                # sampling positions with sales: (3, n_samples, 1, 2)
                coord_mat, coord_vec = self._setup_coordinates(n_samples, pts, scale)
                coord_mats.append(coord_mat)
                coord_vecs.append(coord_vec)

            # apply kernel: 3 x (1, n_scales, ch, n_kernels//n_scales, h, w)
            mats, vecs = self._convolve_with_kernels(
                self.app_mats, self.app_mat_kernels,
                self.app_vecs, self.app_vec_kernels,
                self.app_channels, n_scales
            )
        else:
            # sampling positions without scales: (3, n_samples, 1, 2)
            coord_mats, coord_vecs = self._setup_coordinates(n_samples, pts)

            # 3 x (1, ch, h, w)
            mats, vecs = self.app_mats, self.app_vecs

        # grid sample
        app_features = []
        for i in range(3):
            if self.apply_kernel is True and scales is not None:
                for j in range(n_scales):
                    mat = F.grid_sample(mats[i][:, j], coord_mats[j][[i]], align_corners=True).view(-1, n_samples)
                    vec = F.grid_sample(vecs[i][:, j], coord_vecs[j][[i]], align_corners=True).view(-1, n_samples)
                    if j == 0:
                        app_features.append(mat * vec)
                    else:
                        app_features[-1] = app_features[-1] + (mat * vec)
                app_features[-1] = app_features[-1] / n_scales
            else:
                mat = F.grid_sample(mats[i], coord_mats[[i]], align_corners=True).view(-1, n_samples)
                vec = F.grid_sample(vecs[i], coord_vecs[[i]], align_corners=True).view(-1, n_samples)
                app_features.append(mat * vec)
        
        return self.basis_mat(torch.cat(app_features).T)
        
    @torch.no_grad()
    def upsample_volume_grid(self, grid_size):
        print("=====> UPSAMPLING FEATURE GRID ...")
        def upsample_VM(mats, vecs):
            for i in range(3):
                v0 = self.vec_mode[i]
                m0, m1 = self.mat_mode[i]
                mats[i] = nn.Parameter(F.interpolate(
                    mats[i].data, size=(grid_size[m1], grid_size[m0]),
                    mode="bilinear", align_corners=True
                ))
                vecs[i] = nn.Parameter(F.interpolate(
                    vecs[i].data, size=(grid_size[v0], 1),
                    mode="bilinear", align_corners=True
                ))
            return mats, vecs
        
        self.den_mats, self.den_vecs = upsample_VM(self.den_mats, self.den_vecs)
        self.app_mats, self.app_vecs = upsample_VM(self.app_mats, self.app_vecs)
        self.update_step_size(grid_size)
    
    @torch.no_grad()
    def shrink_volume_grid(self, new_aabb):
        print("=====> SHRINKING FEATURE GRID ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(t_l).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)

        def shrink_VM(mats, vecs):
            for i in range(3):
                v = self.vec_mode[i]
                m0, m1 = self.mat_mode[i]
                mats[i] = nn.Parameter(mats[i].data[..., t_l[m1]:b_r[m1], t_l[m0]:b_r[m0]])
                vecs[i] = nn.Parameter(vecs[i].data[..., t_l[v]:b_r[v], :])
            return mats, vecs

        self.den_mats, self.den_vecs = shrink_VM(self.den_mats, self.den_vecs)
        self.app_mats, self.app_vecs = shrink_VM(self.app_mats, self.app_vecs)

        new_size = b_r - t_l
        self.aabb = new_aabb
        self.update_step_size(new_size)
