#!/usr/bin/env bash
# Source this file for every model: source /path/to/Project3/activate.sh
# Optional old model arguments remain accepted for existing shell commands.
_bev_model="${1:-bevformer}"
case "$_bev_model" in
    bevformer|bevdet|bevdepth) ;;
    *) echo 'Usage: source activate.sh [bevformer|bevdet|bevdepth]' >&2; unset _bev_model; return 1 ;;
esac
if [[ -n ${VIRTUAL_ENV:-} ]] && declare -F deactivate >/dev/null; then
    deactivate
fi
source /home/isa/miniconda3/etc/profile.d/conda.sh
conda activate bevformer
_bev_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROS's Python 3.12 packages must not enter this Python 3.11 environment.
if [[ -d "$_bev_root/BEVFormer/projects" ]]; then
    export PYTHONPATH="$_bev_root/BEVFormer"
else
    export PYTHONPATH="$_bev_root"
fi
export PYTHONPATH="$CONDA_PREFIX/src/BEVDet:$PYTHONPATH:$_bev_root"
export BEV_MODEL="shared"
export BEVDET_ROOT="$CONDA_PREFIX/src/BEVDet"
export BEVDEPTH_ROOT="$CONDA_PREFIX/src/BEVDepth"
export PYTHONNOUSERSITE=1
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# The minimal Conda toolkit puts headers in targets/; library wheels also
# supply CUDA math headers needed by PyTorch's extension headers.
_bev_include="$CONDA_PREFIX/targets/x86_64-linux/include"
for _bev_dir in "$CONDA_PREFIX"/lib/python3.11/site-packages/nvidia/*/include; do
    [[ -d "$_bev_dir" ]] && _bev_include="$_bev_include:$_bev_dir"
done
export CPATH="$_bev_include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
unset _bev_include _bev_dir _bev_root _bev_model
