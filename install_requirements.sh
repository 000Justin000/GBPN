TORCH=1.6.0
CUDA=cu101

# GCC version 8

pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.5 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==1.7.0

pip install pybind11
pip install ogb

make build_cnetworkx
