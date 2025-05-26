if [ ! -d "Color3DMatch-DAC" ]; then
    git clone https://github.com/SheldonFung98/Color3DMatch-DAC.git
    cd Color3DMatch-DAC && ./archive.sh download
fi

docker compose -f .devcontainer/docker-compose.yml build soar
docker compose -f .devcontainer/docker-compose.yml up -d soar