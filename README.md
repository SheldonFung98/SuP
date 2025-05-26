## Introduction
Coming soon!

## Installation

Please use the following command for installation.

1. Clone repo and create docker container.
```bash
git clone https://github.com/SheldonFung98/SOAR.git
cd SOAR && ./config.sh
```
2. Start the container and run
```bash
./post_config.sh
```


## Pre-trained Weights
coming soon

## Color3DMatch/Color3DLoMatch

### Data preparation

The dataset can be downloaded [here](https://drive.google.com/file/d/1pQEo0086ipWwNrroAk_ybnhKildq4o_j/view?usp=sharing). The data is organized as follows

```text
--dataset
         |--data--train--7-scenes-chess--cloud_bin_0.npy
               |      |               |--...
               |      |--...
               |--test--7-scenes-redkitchen--cloud_bin_0.npy
                      |                    |--...
                      |--...
```

### Training

The training code can be found in `experiments/SOAR`. Use the following command for training:

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Testing

To test your model, use the following command:

```bash
# 3DMatch
CUDA_VISIBLE_DEVICES=0 ./eval.sh epoch benchmark
```

`epoch` is the epoch id, `benchmark` can be replaced by [3DMatch, 3DLoMatch, first, second, third, forth]. Different benchmark has different overlap [>0.3, 0.1-0.3, 0.1-0.15, 0.15-0.2, 0.2-0.25, 0.25-0.3]

We also provide pretrained weights in `weights`, use the following command to test the pretrained weights.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../weights/ckpts.pth.tar --benchmark=3DMatch
CUDA_VISIBLE_DEVICES=0 python eval.py --benchmark=3DMatch --method=lgr
```


## Multi-GPU Training

To perform multi-gpu training, use the following command:

```bash
CUDA_VISIBLE_DEVICES=GPUS python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT trainval.py
# example
CUDA_VISIBLE_DEVICES=0,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port='29501' trainval.py
```

## Citation

```bibtex
Coming soon
```

## Acknowledgements
- [PREDATOR](https://github.com/prs-eth/OverlapPredator)
- [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [ColorPCR](https://github.com/mujc2021/ColorPCR)