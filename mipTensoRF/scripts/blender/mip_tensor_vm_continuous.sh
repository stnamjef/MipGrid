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