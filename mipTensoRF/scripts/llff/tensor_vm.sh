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