import os
import sys
import time
import json
import torch
from opt import parse_arguments
from loss import Loss
from utils import *


def train(
    device,                     # computing device (cpu or cuda)
    log_dir,                    # base directory to save the training log and images rendered
    model_kwargs,               # kwargs for model construction (see TensorBase in tensor_base.py)
    loss_kwargs,                # kwargs for loss construction (see Loss in loss.py)
    dataset_kwargs,             # kwargs for dataset construction (see files under data_loader dir.)
    n_samples,                  # number of points sampled along a ray
    step_ratio,                 # step ratio for sampling points
    ndc_ray,                    # indicates if rays are defined in ndc space
    lr_grid,                    # initial learning rate for grid parameters
    lr_network,                 # initial learning rate for mlp and kernel parameters
    lr_decay_iters,             # when to decay the learning rate (-1: do not decay)
    lr_decay_target_ratio,      # a base value to compute lr_factor = lr_decay_target_ratio^(1/lr_decay_iters)
    lr_upsample_reset,          # whether to reset the lr_scale after upsampling feature grids
    batch_size,                 # the number of rays per batch
    n_iters,                    # total number of training iterations
    n_voxel_init,               # an initial voxel resolution
    n_voxel_final,              # a target voxel resolution
    upsample_list,              # a list of iterations at which to upsample the feature grids
    update_alpha_mask_list,     # a list of iterations at which to update the alpha mask
    vis_every,                  # when to visualize the renderings
    progress_refresh_rate,      # how often refresh the tqdm log
    render_train,               # indicates if a model is tested on the train set
    render_test,                # indicates if a model is tested on the test set
    render_path,                # whether to render images along a path
    ckpt,                       # checkpoints to resume the training
    begin_kernel = None
):    
    # intiailize dataset
    train_data = init_dataset(**dataset_kwargs, split="train", batch_size=batch_size, is_stack=False, n_vis=-1, device=device)
    valid_data = init_dataset(**dataset_kwargs, split="test", batch_size=1, is_stack=True, n_vis=5, device=device)

    assert dataset_kwargs["patch_size"] == 1

    # initialize grid resolution
    cur_grid_size = n_voxels_to_resolution(n_voxel_init, train_data.aabb)
    cur_n_samples = min(n_samples, calc_n_samples(cur_grid_size, step_ratio))
    print(f"  number of samples: {cur_n_samples:,d}")

    # load checkpoint if exists
    if ckpt is not None:
        raise NotImplementedError
    else:  # if not, initialize a new model
        additional_kwargs = {
            "device": device,
            "aabb": train_data.aabb,
            "white_bg": train_data.white_bg,
            "near_far": train_data.near_far,
            "step_ratio": step_ratio,
            "grid_size": cur_grid_size
        }
        model = init_model(**model_kwargs, **additional_kwargs)
        model.update_t_distance_radius(train_data.rays, train_data.radii)

    # set learning rate decay factor (gamma)
    if lr_decay_iters > 0:
        lr_decay_factor = lr_decay_target_ratio**(1/lr_decay_iters)
    else:
        lr_decay_iter = n_iters
        lr_decay_factor = lr_decay_target_ratio**(1/n_iters)

    # initialize optimizer
    optim = torch.optim.Adam(model.get_params(lr_grid, lr_network), betas=(0.9, 0.99))

    # initilize loss
    loss_fn = Loss(model, lr_decay_factor, **loss_kwargs)

    print("=====> LEARNING RATE & DECAY FACTOR:")
    print(f"  lr_decay_target_ratio: {lr_decay_target_ratio}, lr_decay_iter: {lr_decay_iter}")
    print(f"  initial orthogonal reg. weight: {loss_fn.ortho_weight}")
    print(f"  initial L1 reg. weight: {loss_fn.l1_weight}")
    print(f"  initial total variance reg. weight: {loss_fn.tv_weight_den} (density), {loss_fn.tv_weight_app} (appearance)")

    # voxel upsampling schedule (liner in logrithmic space)
    n_voxel_list = torch.round(torch.exp(torch.linspace(
        np.log(n_voxel_init), np.log(n_voxel_final), len(upsample_list) + 1
    ))).to(torch.int64).tolist()[1:]

    # empty cache (necessary?)
    torch.cuda.empty_cache()

    # filter rays outside bbox
    if not ndc_ray:
        train_data.rays, train_data.rgbs, train_data.radii, train_data.lossmults = model.filter_rays(
            train_data.rays, train_data.rgbs, train_data.radii, train_data.lossmults, bbox_only=True
        )
        train_data.reset_sampler()

    # main loop
    begin = time.time()
    psnr_train, psnr_valid = [], None
    pbar = tqdm(range(1, n_iters + 1), position=0, leave=True, miniters=progress_refresh_rate, file=sys.stdout)
    for i in pbar:
        # get a random batch & ship to CUDA
        rays, rgbs_gt, radii, lossmults = train_data.fetch_data()

        # render rays
        rgbs_pred, _ = model(rays, cur_n_samples, True, ndc_ray, radii)

        # compute loss & do backprop
        optim.zero_grad()
        loss_value = loss_fn.accumulate_gradients(i, rgbs_pred, rgbs_gt, lossmults)
        optim.step()

        # learning rate schedule
        decay_learning_rate(optim, lr_decay_factor)

        # save metrics
        psnr_train.append(calc_psnr(loss_value))

        # evaluation
        if i % vis_every == 0:
            psnr_valid = evaluation(model, valid_data, cur_n_samples, ndc_ray, f"{log_dir}/imgs_vis", f"{i:06d}_", False)
        
        # logging
        if i % progress_refresh_rate == 0:
            log = [
                f"Iter: {i}/{n_iters:6d}",
                f"L2: {loss_value:2.6f}",
                f"PSNR_train: {float(np.mean(psnr_train)):2.2f}"
            ]

            if psnr_valid is not None:
                log += [f"PSNR_{k}: {float(v):2.2f}" for k, v in psnr_valid.items()]

                # logger.info(log)
                pbar.set_description(", ".join(log))
                psnr_train = []

        # update alpha mask & shrink grid
        if i in update_alpha_mask_list:
            # alpha volume size
            if torch.prod(cur_grid_size) < 256**3:
                cur_mask_size = cur_grid_size
            else:
                raise ValueError("prod(cur_grid_size) > 256**3")
            
            # update alpha mask
            new_aabb = model.update_alpha_mask(cur_mask_size)

            # shrink feature grid (only at the first alpha update)
            if i == update_alpha_mask_list[0]:
                model.shrink_volume_grid(new_aabb)
                model.update_t_distance_radius(train_data.rays, train_data.radii)
                loss_fn.reset_l1_weight()
            
            # filter rays (only at the second alpha update)
            if not ndc_ray and i == update_alpha_mask_list[1]:
                train_data.rays, train_data.rgbs, train_data.radii, train_data.lossmults = model.filter_rays(
                    train_data.rays, train_data.rgbs, train_data.radii, train_data.lossmults
                )
                train_data.reset_sampler()
            
        # upsample feature grids
        if i in upsample_list:
            # reset grid resolution
            cur_grid_size = n_voxels_to_resolution(n_voxel_list.pop(0), model.aabb)
            cur_n_samples = min(n_samples, calc_n_samples(cur_grid_size, step_ratio))
            print(f"  number of samples: {cur_n_samples}")

            # upsample
            model.upsample_volume_grid(cur_grid_size)

            # reset learning rate and optimizer
            lr_scale = 1 if lr_upsample_reset else lr_decay_target_ratio**(i/n_iters)
            optim = torch.optim.Adam(model.get_params(lr_grid*lr_scale, lr_network*lr_scale), betas=(0.9, 0.99))
        
        # begin kernel training
        if i == begin_kernel:
            print("=====> BEGIN KERNEL TRAINING ...")
            model.init_kernels()

            # reset learning rate and optimizer
            optim = torch.optim.Adam(model.get_params(lr_grid, lr_network), betas=(0.9, 0.99))
    
    # record training time
    time_elapsed = time.time() - begin
    minutes = int(time_elapsed // 60)
    seconds = time_elapsed % 60
    print(f"=====> Training took {time_elapsed:.2f} sec. ({minutes:d} min. {seconds:.2f} sec.)")

    # total number of parameters
    total_params = sum([p.numel() for params_dict in model.get_params(lr_grid, lr_network) for p in params_dict["params"]])
    print(f"=====> Total number of parameters: {total_params:,d}")

    # save training time and # params
    np.savetxt(os.path.join(log_dir, "time.txt"), [time_elapsed, total_params])

    # save the last model
    model.n_samples = cur_n_samples
    model.save(os.path.join(log_dir, "params.th"))

    del train_data, valid_data

    if render_train:
        save_dir = f"{log_dir}/imgs_train_all"
        os.makedirs(save_dir, exist_ok=True)
        train_data = init_dataset(**dataset_kwargs, split="train", batch_size=1, is_stack=True, n_vis=-1, device=device)
        psnr_train = evaluation(model, train_data, cur_n_samples, ndc_ray, save_dir)
        log = ", ".join([f"PSNR_{k}: {float(v):2.2f}" for k, v in psnr_train.items()])
        print(f'============> evaluation on train set: {log} <============')
    
    if render_test:
        save_dir = f"{log_dir}/imgs_test_all"
        os.makedirs(save_dir, exist_ok=True)
        test_data = init_dataset(**dataset_kwargs, split="test", batch_size=1, is_stack=True, n_vis=-1, device=device)
        psnr_test = evaluation(model, test_data, cur_n_samples, ndc_ray, save_dir)
        log = ", ".join([f"psnr_{k}: {float(v):2.2f}" for k, v in psnr_test.items()])
        print(f'============> evaluation on test set: {log} <============')
    
    if render_path:
        raise NotImplementedError


@torch.no_grad()
def test(
    device,                     # computing device (cpu or cuda)
    log_dir,                    # base directory to save the training log and images rendered
    model_kwargs,               # kwargs for model construction (see TensorBase in tensor_base.py)
    dataset_kwargs,             # kwargs for dataset construction (see files under data_loader dir.)
    ndc_ray,                    # indicates if rays are defined in ndc space
    render_train,               # indicates if a model is tested on the train set
    render_test,                # indicates if a model is tested on the test set
    render_path,                # whether to render images along a path
    ckpt,                       # checkpoints to resume the training
):
    # load checkpoint
    checkpoint = torch.load(ckpt, map_location=device)
    checkpoint["kwargs"].update({"device": device})

    # initialize model
    model = eval(model_kwargs["name"])(**checkpoint["kwargs"])
    model.load(checkpoint)

    if render_train:
        save_dir = f"{log_dir}/imgs_train_all"
        os.makedirs(save_dir, exist_ok=True)
        train_data = init_dataset(**dataset_kwargs, split="train", batch_size=1, is_stack=True, n_vis=-1, device=device)
        psnr_train = evaluation(model, train_data, -1, ndc_ray, save_dir)
        log = ", ".join([f"PSNR_{k}: {float(v):2.2f}" for k, v in psnr_train.items()])
        print(f'============> evaluation on train set: {log} <============')
    
    if render_test:
        save_dir = f"{log_dir}/imgs_test_all"
        os.makedirs(save_dir, exist_ok=True)
        test_data = init_dataset(**dataset_kwargs, split="test", batch_size=1, is_stack=True, n_vis=-1, device=device)
        psnr_test = evaluation(model, test_data, -1, ndc_ray, save_dir)
        log = ", ".join([f"psnr_{k}: {float(v):2.2f}" for k, v in psnr_test.items()])
        print(f'============> evaluation on test set: {log} <============')
    
    if render_path:
        raise NotImplementedError


if __name__ == "__main__":
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse arguments
    args, cfg = parse_arguments()

    # set seed and number of threads
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_threads > 0:
        torch.set_num_threads(args.n_threads)

    # logging directory
    log_dir = f"{args.base_dir}/{args.exp_name}"

    # check if a model name is valid
    assert cfg.model["name"] in ["TensorVM", "MipTensorVM", "MipTensorVMMultiscaleKernel"]

    # begin main program
    if args.render_only:
        # print config
        del cfg.training
        print("TESTING CONFIG")
        print(json.dumps(cfg, indent=2))
        
        # begin test
        if args.render_arbitrary_resolution:
            raise NotImplementedError
            # test_arbitrary_resolution(device, log_dir, cfg.model, cfg.dataset, **cfg.testing)
        else:
            test(device, log_dir, cfg.model, cfg.dataset, **cfg.testing)
    else:
        # make directory
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f"{log_dir}/imgs_vis", exist_ok=True)

        # save config
        with open(os.path.join(log_dir, "config.json"), "wt") as f:
            json.dump(cfg, f, indent=2)
        
        # print config
        print("TRAINING CONFIG:")
        print(json.dumps(cfg, indent=2))

        # begin training
        train(device, log_dir, cfg.model, cfg.loss, cfg.dataset, **cfg.training)
