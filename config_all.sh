#!/bin/bash
# This script is used to set up the environment in the docker container.

if [ ! -d "weights" ]; then
    mkdir -p weights
    wget -O weights/weights.pth.tar https://github.com/mujc2021/ColorPCR/releases/download/ckpts/weights.pth.tar
fi

if [ ! -d "Color3DMatch-DAC" ]; then
    git clone https://github.com/SheldonFung98/Color3DMatch-DAC.git
    cd Color3DMatch-DAC && ./archive.sh download
    cd ..
fi

EXT=ext.cpython-310-x86_64-linux-gnu.so
if [ ! -f "$EXT" ]; then
    python setup.py build develop
fi

if [ ! -d "dataset" ]; then
    ./Color3DMatch-DAC/archive.sh link dataset
fi

