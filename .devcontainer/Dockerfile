# Use the official PyTorch image with CUDA and Python preinstalled
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set a working directory
WORKDIR /workspace

# Set environment variables to avoid prompts from tools
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    unzip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    x11-apps \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Create and activate a virtual environment
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN git clone https://github.com/SheldonFung98/PointScope.git
RUN cd PointScope && pip3 install -e .

# (Optional) JupyterLab
## install JupyterLab
# RUN pip install jupyterlab && \
#     jupyter notebook --generate-config && \
#     echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
#     echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
#     echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py

# ## Expose port for JupyterLab
# EXPOSE 8888
# ## Default command
# CMD [ "jupyter", "lab", "--allow-root", "--notebook-dir=/workspace" ]
