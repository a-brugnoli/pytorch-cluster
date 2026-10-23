ssh <username>@pando
module load python/2023-3
export http_proxy=http://proxy.isae.fr:3128
# Create and activate the Conda environment
conda create -n pytorch-wheel python=3.11 pip -y
conda activate pytorch-wheel

# Install PyTorch 2.5.1 with CUDA 11.8
python -m pip install torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install NumPy compatible with the cluster's older Linux system
python -m pip install --only-binary=:all: "numpy==1.26.4"