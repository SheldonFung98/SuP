# SuP: Sub-cloud Driven Point Cloud Registration

<p align="left">
  <a href="https://cvpr.thecvf.com/Conferences/2026"><img src="https://img.shields.io/badge/CVPR%202026-Highlight-9b1c1c?style=for-the-badge&labelColor=2b2b2b" alt="CVPR 2026 Highlight"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.7%2B-76b900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
</p>

> **SuP: Sub-cloud Driven Point Cloud Registration**
>
> Sheldon&nbsp;Fung¹, Wei&nbsp;Pan², Ling&nbsp;Cao², Fei&nbsp;Hou³,⁴, Ling&nbsp;Chen⁵, Shasha&nbsp;Mao⁶, Hongdong&nbsp;Li⁷, Xuequan&nbsp;Lu¹⁎
>
> ¹University of Western Australia · ²OPT Machine Vision · ³Institute of Software, CAS · ⁴University of Chinese Academy of Sciences · ⁵University of Technology Sydney · ⁶Xidian University · ⁷Australian National University
>
> ⁎ Corresponding author.

📄 **Paper:** *coming soon* &nbsp;|&nbsp; 📦 **Code:** this repository &nbsp;|&nbsp; 🗂️ **Data (C3DM / C3DLM):** [Google&nbsp;Drive](https://drive.google.com/file/d/1pQEo0086ipWwNrroAk_ybnhKildq4o_j/view?usp=sharing)

---

## Introduction

Existing learning-based point-cloud-registration methods handle high-overlap pairs well but **struggle when the overlap is low**, because geometric or semantic similarities in the *non-overlapping* regions inevitably produce ambiguous matches.

**SuP** reformulates low-overlap registration as a **high-overlap sub-cloud anchor pair mining** problem. The core component is the **Dual-phase Sub-cloud Anchor Mining (DSAM)** module:

1. **Subdivide** the source and target point clouds into multiple sub-clouds.
2. **Phase 1 — OPS** *(Overlap-guided Prior-weighting Scheme)*: leverages feature salience to cheaply pre-score candidate sub-cloud anchor pairs.
3. **Phase 2 — MPN** *(Multi-scale Post-weighting Network)*: refines the ranking by exploiting neighborhood feature consensus at multiple scales.
4. **Merge-to-match**: the top-ranked anchor pairs are merged to produce final dense correspondences, from which the transformation is recovered via either **LGR** (RANSAC-free) or **RANSAC**.

DSAM is supervised end-to-end by an **alignment-aware weighting loss (AWL)** that uses on-the-fly anchor-pair alignment errors as the ranking target.

## Main results

Registration Recall (%) on **Color3DMatch (C3DM)** and **Color3DLoMatch (C3DLM)**:

| Method | Estimator | C3DM RR ↑ | C3DLM RR ↑ |
|---|:---:|:---:|:---:|
| CoFiNet | RANSAC-50k | 89.3 | 67.5 |
| GeoTransformer | RANSAC-50k | 92.0 | 75.0 |
| PEAL | RANSAC-50k | 94.6 | 81.7 |
| ColorPCR | RANSAC-50k | 96.7 | 88.9 |
| **SuP (ours)** | **RANSAC-50k** | **98.1** | **90.4** |
| CoFiNet | LGR | 87.6 | 64.8 |
| GeoTransformer | LGR | 91.5 | 74.0 |
| PEAL | LGR | 94.3 | 81.2 |
| ColorPCR | LGR | 96.5 | 88.3 |
| **SuP (ours)** | **LGR** | **97.8** | **90.2** |

SuP is the new state of the art on both benchmarks under both estimators; the LGR (RANSAC-free) numbers also match or exceed prior RANSAC-50k baselines.

## Installation

SuP is tested with PyTorch ≥ 1.13 + CUDA ≥ 11.7 and builds two custom C++/CUDA extensions for point-cloud subsampling and radius neighbor search.

### Ubuntu / Linux

```bash
git clone https://github.com/SheldonFung98/SuP.git
cd SuP

# (optional) create a fresh environment
conda create -n sup python=3.10 -y
conda activate sup

# install PyTorch matching your CUDA toolkit (1.13+ recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# project dependencies
pip install -r requirements.txt

# build the C++/CUDA extensions
python setup.py build_ext --inplace
```

The Docker workflow (`./config.sh` + `./post_config.sh`) used during development is also supported on Linux.

### Windows

The repo now compiles cleanly under MSVC (Visual Studio 2019 / 2022). Prerequisites:

- Visual Studio **Build Tools 2019 or 2022** with the *"Desktop development with C++"* workload.
- CUDA Toolkit matching your PyTorch build (e.g. CUDA 11.8 → PyTorch 2.0+cu118).
- Python 3.10 (Anaconda recommended).

From a *"x64 Native Tools Command Prompt for VS 2022"* or a PowerShell with `vcvarsall.bat` sourced:

```powershell
git clone https://github.com/SheldonFung98/SuP.git
cd SuP

conda create -n sup python=3.10 -y
conda activate sup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# build C++/CUDA extensions (uses MSVC + nvcc; setup.py auto-selects /O2 /std:c++17 /EHsc on Windows)
python setup.py build_ext --inplace
```

If you see `LNK2019 unresolved external symbol` from `torch_python.lib`, make sure the PyTorch wheel matches your Python version and that `cl.exe` is on `PATH` (i.e. you launched from the VS Native Tools shell).

## Pre-trained weights

Coming soon — will be released here as a GitHub Release.

## Color3DMatch / Color3DLoMatch

### Data layout

Download the dataset [here](https://drive.google.com/file/d/1pQEo0086ipWwNrroAk_ybnhKildq4o_j/view?usp=sharing) and arrange it as:

```text
dataset/
├── data/
│   ├── train/7-scenes-chess/cloud_bin_0.npy
│   │   └── ...
│   └── test/7-scenes-redkitchen/cloud_bin_0.npy
│       └── ...
```

### Training

The training entry point lives in `experiments/SOAR`:

```bash
# single GPU
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Testing

```bash
# 3DMatch
CUDA_VISIBLE_DEVICES=0 ./eval.sh <epoch> <benchmark>
```

`<epoch>` is the checkpoint epoch id; `<benchmark>` is one of `3DMatch`, `3DLoMatch`, `first`, `second`, `third`, `forth`. The latter four correspond to overlap bins `[0.1–0.15, 0.15–0.2, 0.2–0.25, 0.25–0.3]`; the first two cover `>0.3` (C3DM) and `0.1–0.3` (C3DLM) respectively.

To evaluate a released checkpoint directly:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../weights/ckpts.pth.tar --benchmark=3DMatch
CUDA_VISIBLE_DEVICES=0 python eval.py --benchmark=3DMatch --method=lgr
```

### Multi-GPU training

```bash
CUDA_VISIBLE_DEVICES=<GPUS> python -m torch.distributed.launch \
    --nproc_per_node=<N_GPU> --master_port=<PORT> trainval.py

# example: 2 GPUs
CUDA_VISIBLE_DEVICES=0,5 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29501 trainval.py
```

## Citation

If you find SuP useful, please cite:

```bibtex
@inproceedings{fung2026sup,
  title     = {SuP: Sub-cloud Driven Point Cloud Registration},
  author    = {Fung, Sheldon and Pan, Wei and Cao, Ling and Hou, Fei and
               Chen, Ling and Mao, Shasha and Li, Hongdong and Lu, Xuequan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  note      = {Highlight}
}
```

## Acknowledgements

SuP builds on the excellent work of:

- [PREDATOR](https://github.com/prs-eth/OverlapPredator)
- [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [ColorPCR](https://github.com/mujc2021/ColorPCR)
