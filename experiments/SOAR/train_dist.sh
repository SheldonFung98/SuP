GPU_NUM=4
torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPU_NUM \
    trainval.py --snapshot ../../weights/weights.pth.tar