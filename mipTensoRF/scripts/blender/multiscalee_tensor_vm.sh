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