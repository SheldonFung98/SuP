if [ ! -d "weights" ]; then
    mkdir -p weights
    wget -O weights/weights.pth.tar https://github.com/mujc2021/ColorPCR/releases/download/ckpts/weights.pth.tar
fi

if [ ! -d "Color3DMatch-DAC" ]; then
    git clone https://github.com/SheldonFung98/Color3DMatch-DAC.git
    cd Color3DMatch-DAC && ./archive.sh download
    cd ..
fi

docker compose -f .devcontainer/docker-compose.yml build soar
docker compose -f .devcontainer/docker-compose.yml up -d soar