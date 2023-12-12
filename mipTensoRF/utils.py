import os
import cv2
import scipy
import imageio
import torch
import numpy as np
from tqdm import tqdm
from data_loader import dataset_dict
from models.tensor_vm import TensorVM, MipTensorVM


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    return cv2.applyColorMap(x, cmap)


def calc_psnr(mse):
    return -10. * np.log(mse) / np.log(10.)


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


def render_minibatch(model, rays, radii, n_samples, ndc_ray, chunk=16384):
    n_rays, rgbs, depths = len(rays), [], []
    for chunk_idx in range(n_rays//chunk + int(n_rays%chunk > 0)):
        rays_chunk = rays[chunk_idx*chunk:(chunk_idx+1)*chunk]
        if radii is not None:
            radii_chunk = radii[chunk_idx*chunk:(chunk_idx+1)*chunk]
        else:
            radii_chunk = None
        rgb, depth = model(rays_chunk, n_samples, False, ndc_ray, radii_chunk)
        rgbs.append(rgb)
        depths.append(depth)
    return torch.cat(rgbs), torch.cat(depths)


@torch.no_grad()
def evaluation(model, dataset, n_samples, ndc_ray, save_dir, prtx='', extra_metrics=True):
    # make sure all the datas are stacked
    assert dataset.is_stack == True

    # eval_function for a single image
    def eval_fn(rays, rgbs_gt, radii):
        # total metrics
        metrics = []

        # height and width of the gt image
        height, width, _ = rgbs_gt.shape

        # render rays
        rgbs, _ = render_minibatch(model, rays, radii, n_samples, ndc_ray)

        # reshape
        rgbs = rgbs.clamp(0., 1.).reshape(height, width, 3)
        # depths = depths.reshape(height, width).cpu()

        # calculate MSE and PSNR
        loss = torch.mean((rgbs - rgbs_gt)**2)
        metrics.append(calc_psnr(loss.item()))

        # ship back to cpu
        rgbs = rgbs.cpu().numpy()
        rgbs_gt = rgbs_gt.cpu().numpy()

        # caculate SSIM, LPIPS (VGG)
        if extra_metrics:
            metrics.append(rgb_ssim(rgbs, rgbs_gt, 1))
            metrics.append(rgb_lpips(rgbs_gt, rgbs, "vgg", model.device))

        return metrics, (rgbs * 255).astype(np.uint8)

    # make save directory
    os.makedirs(save_dir, exist_ok=True)

    try:
        tqdm._instance.clear()
    except Exception:
        pass

    # begin evaluation
    metrics_all = []
    for scale_idx in range(dataset.n_scales):
        metrics_scale = []
        for image_idx in tqdm(range(dataset.n_images), position=0, leave=True, desc=f"scale{scale_idx}"):
            rays, rgbs_gt, radii, _ = dataset.fetch_data(scale_idx * dataset.n_images + image_idx)
            metrics, image = eval_fn(rays, rgbs_gt, radii)
            metrics_scale.append(metrics)
            imageio.imwrite(os.path.join(save_dir, f"{prtx}{image_idx:03d}_d{scale_idx}.png"), image)
        metrics_all.append(np.mean(metrics_scale, axis=0))
    
    # aggregate only PSNR
    PSNRs = {scale_idx: metrics[0] for scale_idx, metrics in enumerate(metrics_all)}

    # save metrics aggregated
    np.savetxt(f"{save_dir}/{prtx}metrics_all.txt", metrics_all)

    if dataset.n_scales > 1:
        # append avg. PSNR
        PSNRs["avg"] = np.mean(list(PSNRs.values()))
    
    return PSNRs


def init_dataset(name, split, batch_size, is_stack, n_vis, device, **dataset_kwargs):
    if "blender" in name.split('_'):
        return dataset_dict[name](**dataset_kwargs, split=split, batch_size=batch_size, is_stack=is_stack, n_vis=n_vis, device=device)
    else:
        return dataset_dict[name](**dataset_kwargs, split=split, batch_size=batch_size, is_stack=is_stack, n_vis=n_vis, device=device, hold_every=8)

def init_model(name, **kwargs):
    return eval(name)(**kwargs)


def n_voxels_to_resolution(n_voxels, aabb):
    assert aabb.ndim == 2 and aabb.shape[-1] == 3
    xyz_min, xyz_max = aabb
    voxel_size = (torch.prod(xyz_max - xyz_min) / n_voxels)**(1/3)
    return ((xyz_max - xyz_min) / voxel_size).long()


def calc_n_samples(grid_size, step_ratio):
    return int(torch.norm(grid_size.float()) / step_ratio)


def decay_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * lr_decay


# @torch.no_grad()
# def test_arbitrary_resolution(
#     device,                     # computing device (cpu or cuda)
#     log_dir,                    # base directory to save the training log and images rendered
#     model_kwargs,               # kwargs for model construction (see TensorBase in tensor_base.py)
#     dataset_kwargs,             # kwargs for dataset construction (see files under data_loader dir.)
#     ndc_ray,                    # indicates if rays are defined in ndc space
#     render_train,               # indicates if a model is tested on the train set
#     render_test,                # indicates if a model is tested on the test set
#     render_path,                # whether to render images along a path
#     ckpt,                       # checkpoints to resume the training
# ):
#     assert dataset_kwargs["name"] == "blender"  # support only blender dataset
#     assert ndc_ray == False

#     # replace downsample scale with multiple scales
#     del dataset_kwargs["downsample"]
#     downsamples = [8/7., 8/6., 8/5., 8/4., 8/3., 8/2., 8/1.]

#     def eval_fn_arbitrary_resolution(model, split, save_dir):
#         psnrs = {}
#         for down in downsamples:
#             # initialize dataset
#             _, _, _, rays, rgbs, radii, _ = init_dataset(**dataset_kwargs, split=split, downsample=down, is_stack=True)
#             # begin evaluation
#             psnr = evaluation(model, rays, rgbs, radii, -1, ndc_ray, -1, save_dir)
#             # update dictionary
#             psnrs.update(psnr)
#         log = ", ".join([f'psnr_{k}: {float(v):2.2f}' for k, v in psnrs.items()])
#         print(f'============> evaluation on {split} set: {log} <============')
#         # aggregate_results(save_dir, split=f"{split}_arbitrary_resolution")

#     # load checkpoint
#     checkpoint = torch.load(ckpt, map_location=device)
#     checkpoint["kwargs"].update({"device": device, "learnable_kernel": True})

#     # initialize model
#     model = eval(model_kwargs["name"])(**checkpoint["kwargs"])
#     model.load(checkpoint)

#     # begin test
#     if render_train:
#         save_dir = f"{log_dir}/imgs_train_arbitrary_resolution"
#         os.makedirs(save_dir, exist_ok=True)
#         eval_fn_arbitrary_resolution(model, "train", save_dir)
    
#     if render_test:
#         save_dir = f"{log_dir}/imgs_test_arbitrary_resolution"
#         os.makedirs(save_dir, exist_ok=True)
#         eval_fn_arbitrary_resolution(model, "test", save_dir)

#     if render_path:
#         raise NotImplementedError
