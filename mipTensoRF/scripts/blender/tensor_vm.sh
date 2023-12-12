# training
for data in chair drums ficus hotdog lego materials mic ship
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config=configs/blender/tensor_vm.yaml \
        --exp_name="blender_tensor_vm_40000_"$data \
        --dataset.data_dir=/workspace/dataset/nerf_synthetic/$data \
        --training.n_iters=40000
done

# test on the multi-scale dataset
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