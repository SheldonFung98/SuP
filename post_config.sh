#!/bin/bash
# This script is used to set up the environment in the docker container.

EXT=ext.cpython-310-x86_64-linux-gnu.so
if [ ! -f "$EXT" ]; then
    python setup.py build develop
fi

if [ ! -d "dataset" ]; then
    ./Color3DMatch-DAC/archive.sh link dataset
fi

