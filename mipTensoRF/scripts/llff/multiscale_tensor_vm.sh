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