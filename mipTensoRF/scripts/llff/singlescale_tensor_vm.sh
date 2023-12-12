for data in fern flower fortress horns leaves orchids room trex
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/llff/mip_tensor_vm.yaml \
        --exp_name="llff_singlescale_tensor_vm_25000_"$data \
        --dataset.data_dir=/workspace/dataset/nerf_llff_data/$data \
        --training.n_iters=25000 \
        --training.begin_kernel=1000000000 # 1e9 (do not use kernel)
done