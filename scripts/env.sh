# Source me: keeps all caches/temp under the wakeword folder (wherever it lives: WW is this
# file's parent folder)
export WW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=$WW/cache/hf
export UV_CACHE_DIR=$WW/cache/uv
export PIP_CACHE_DIR=$WW/cache/pip
export UV_PYTHON_INSTALL_DIR=$WW/cache/uv-python
export XDG_CACHE_HOME=$WW/cache/xdg
export TMPDIR=$WW/tmp
export TORCH_HOME=$WW/cache/torch
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export HF_HUB_DISABLE_TELEMETRY=1
