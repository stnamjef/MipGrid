for data in chair drums ficus hotdog lego materials mic ship
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/blender/mip_tensor_vm.yaml \
        --exp_name="blender_singlescale_tensor_vm_40000_"$data \
        --dataset.data_dir=/workspace/dataset/nerf_synthetic/$data \
        --training.n_iters=40000 \
        --training.begin_kernel=1000000000 # 1e9 (do not use kernel)
done
