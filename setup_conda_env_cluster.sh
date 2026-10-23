ssh <username>@pando
module load python/2023-3
export http_proxy=http://proxy.isae.fr:3128
conda create -n pytorch-env python=3.11 -y
