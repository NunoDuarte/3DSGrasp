FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV GRASP_ROOT=/workspace/3DSGrasp

WORKDIR $GRASP_ROOT

# Install required system dependencies
# Remove old NVIDIA repository lists to prevent GPG key expired errors
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y \
    git \
    vim \ 
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy only the Completion folder
COPY Completion/ ./Completion/

# Install the Python dependencies for the Completion Network
# Note: Since the original requirements.txt is truncated and contains unrelated
# ROS dependencies, we install the specific packages required by the Completion network.
RUN pip install --upgrade pip && \
    pip install \
    open3d==0.15.2 \
    numpy \
    PyYAML \
    matplotlib \
    tensorboardX \
    scipy \
    tqdm \
    gdown \
    easydict \
    timm

# Install deep learning 3D operators
# Pointnet2_PyTorch and KNN_CUDA. We patch KNN_CUDA to bypass the
# torch.cuda.is_available() check which fails during docker build.
RUN export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX" && \
    pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib" && \
    git clone https://github.com/unlimblue/KNN_CUDA.git /tmp/KNN_CUDA && \
    cd /tmp/KNN_CUDA && \
    sed -i 's/assert torch.cuda.is_available()/pass #/' knn_cuda/__init__.py && \
    pip install . && \
    rm -rf /tmp/KNN_CUDA

# Download the pre-trained model from Google Drive
RUN mkdir -p /models && \
    gdown --id 11vTsY0MQw9pzsqz3MyvCKjQT2rQ9VxVi -O /models/3dsgrasp_model.pth

# Download and extract the training/testing dataset
RUN mkdir -p /data && cd /data && \
    gdown --id 1rnJP3Q2zvcj5uImxRu8yYwgk0O7md8dJ -O dataset.zip && \
    unzip dataset.zip && \
    rm dataset.zip && \
    (mv */input . 2>/dev/null || true) && \
    (mv */gt . 2>/dev/null || true)

# Install the custom chamfer_dist CUDA extension
# Set TORCH_CUDA_ARCH_LIST to avoid the "list index out of range" error
# because the GPU architecture cannot be auto-detected during docker build
RUN cd ./Completion/extensions/chamfer_dist && \
    export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX" && \
    python setup.py install

# Set working directory to Completion
WORKDIR $GRASP_ROOT/Completion

# Default command
CMD ["/bin/bash"]

