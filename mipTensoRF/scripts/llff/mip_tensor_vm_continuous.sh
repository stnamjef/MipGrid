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