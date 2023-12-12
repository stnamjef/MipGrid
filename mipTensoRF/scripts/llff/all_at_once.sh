# training
for data in fern flower fortress horns leaves orchids room trex
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/llff/tensor_vm.yaml \
        --exp_name="llff_tensor_vm_25000_"$data \
        --dataset.data_dir=/workspace/dataset/nerf_llff_data/$data \
        --training.n_iters=25000
done

# test on the multi-scale dataset
for data in fern flower fortress horns leaves orchids room trex
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/llff/tensor_vm.yaml \
        --exp_name="llff_tensor_vm_25000_"$data \
        --render_only \
        --dataset.name=multiscale_llff \
        --dataset.n_downsamples=4 \
        --dataset.return_radii \
        --dataset.data_dir=/workspace/dataset/nerf_llff_data/$data \
        --testing.ckpt=/workspace/mipTensoRF/log/llff_tensor_vm_25000_$data/params.th
done

# singlescale TensoRF
for data in fern flower fortress horns leaves orchids room trex
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/llff/mip_tensor_vm.yaml \
        --exp_name="llff_singlescale_tensor_vm_25000_"$data \
        --dataset.data_dir=/workspace/dataset/nerf_llff_data/$data \
        --training.n_iters=25000 \
        --training.begin_kernel=1000000000 # 1e9 (do not use kernel)
done

# multiscale TensoRF
for data in fern flower fortress horns leaves orchids room trex
do
    for downsample in 1.0 2.0 4.0 8.0
    do
        CUDA_VISIBLE_DEVICES=0 python main.py \
            --config=configs/llff/tensor_vm.yaml \
            --exp_name="llff_multiscale_tensor_vm_25000_"$downsample"_"$data \
            --dataset.downsample=$downsample \
            --dataset.data_dir=/workspace/dataset/nerf_llff_data/$data \
            --training.n_iters=25000
    done
done

# discrete scale coordinate
for data in fern flower fortress horns leaves orchids room trex
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/llff/mip_tensor_vm.yaml \
        --exp_name="llff_mip_tensor_vm_discrete_25000_"$data \
        --model.kernel_size=3 \
        --model.scale_types="cylinder_radius" \
        --model.learnable_kernel \
        --dataset.data_dir=/workspace/dataset/nerf_llff_data/$data \
        --training.n_iters=25000 \
        --training.begin_kernel=5500
done

# continuous scale coordinate
for data in fern flower fortress horns leaves orchids room trex
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/llff/mip_tensor_vm.yaml \
        --exp_name="llff_mip_tensor_vm_continuous_25000_"$data \
        --model.kernel_size=3 \
        --model.scale_types="cone_radius" \
        --model.learnable_kernel \
        --dataset.data_dir=/workspace/dataset/nerf_llff_data/$data \
        --training.n_iters=25000 \
        --training.begin_kernel=5500
done

# 2D scale coordiante
for data in fern flower fortress horns leaves orchids room trex
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/llff/mip_tensor_vm.yaml \
        --exp_name="llff_mip_tensor_vm_2d_25000_"$data \
        --model.kernel_size=3 \
        --model.n_kernels=8 \
        --model.scale_types="distance" \
        --model.scale_types="cone_radius" \
        --model.learnable_kernel \
        --dataset.data_dir=/workspace/dataset/nerf_llff_data/$data \
        --training.n_iters=25000 \
        --training.begin_kernel=5500
done