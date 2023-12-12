# tensor-vm training
for data in chair drums ficus hotdog lego materials mic ship
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/blender/tensor_vm.yaml \
        --exp_name="blender_tensor_vm_40000_"$data \
        --dataset.data_dir=/workspace/dataset/nerf_synthetic/$data \
        --training.n_iters=40000
done

# tensor-vm test on the multi-scale dataset
for data in chair drums ficus hotdog lego materials mic ship
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/blender/tensor_vm.yaml \
        --exp_name="blender_tensor_vm_40000_"$data \
        --render_only \
        --dataset.name=multiscale_blender \
        --dataset.n_downsamples=4 \
        --dataset.return_radii \
        --dataset.data_dir=/workspace/dataset/nerf_synthetic/$data \
        --testing.ckpt=/workspace/mipTensoRF/log/blender_tensor_vm_40000_$data/params.th
done

# singlescale TensoRF
for data in chair drums ficus hotdog lego materials mic ship
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/blender/mip_tensor_vm.yaml \
        --exp_name="blender_singlescale_tensor_vm_40000_"$data \
        --dataset.data_dir=/workspace/dataset/nerf_synthetic/$data \
        --training.n_iters=40000 \
        --training.begin_kernel=1000000000 # 1e9 (do not use kernel)
done

# multiscale TensoRF
for data in chair drums ficus hotdog lego materials mic ship
do
    for downsample in 1.0 2.0 4.0 8.0
    do
        CUDA_VISIBLE_DEVICES=0 python main.py \
            --config=configs/blender/tensor_vm.yaml \
            --exp_name="blender_multiscale_tensor_vm_40000_"$downsample"_"$data \
            --dataset.downsample=$downsample \
            --dataset.data_dir=/workspace/dataset/nerf_synthetic/$data \
            --training.n_iters=40000
    done
done

# discrete scale coordinate
for data in chair drums ficus hotdog lego materials mic ship
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/blender/mip_tensor_vm.yaml \
        --exp_name="blender_mip_tensor_vm_discrete_40000_"$data \
        --model.kernel_size=3 \
        --model.scale_types="cylinder_radius" \
        --model.learnable_kernel \
        --dataset.data_dir=/workspace/dataset/nerf_synthetic/$data \
        --training.n_iters=40000 \
        --training.begin_kernel=7000
done

# continuous scale coordinate
for data in chair drums ficus hotdog lego materials mic ship
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/blender/mip_tensor_vm.yaml \
        --exp_name="blender_mip_tensor_vm_continuous_40000_"$data \
        --model.kernel_size=3 \
        --model.scale_types="cone_radius" \
        --model.learnable_kernel \
        --dataset.data_dir=/workspace/dataset/nerf_synthetic/$data \
        --training.n_iters=40000 \
        --training.begin_kernel=7000
done

# 2D scale coordiante
for data in chair drums ficus hotdog lego materials mic ship
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/blender/mip_tensor_vm.yaml \
        --exp_name="blender_mip_tensor_vm_2d_40000_"$data \
        --model.kernel_size=3 \
        --model.n_kernels=8 \
        --model.scale_types="distance" \
        --model.scale_types="cone_radius" \
        --model.learnable_kernel \
        --dataset.data_dir=/workspace/dataset/nerf_synthetic/$data \
        --training.n_iters=40000 \
        --training.begin_kernel=7000
done