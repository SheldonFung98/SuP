
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPU_NUM \
    trainval.py --snapshot ../../weights/weights.pth.tar
