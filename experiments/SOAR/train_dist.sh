
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
torchrun --nproc_per_node=$GPU_NUM trainval.py --snapshot ../../weights/weights.pth.tar
