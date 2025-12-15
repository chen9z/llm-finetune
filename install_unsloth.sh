# Blackwell GPU in Winddows WSL2
export TORCH_CUDA_ARCH_LIST="12.0"

uv pip install unsloth
# use torch cu128
uv pip install -U vllm --torch-backend=cu128
uv pip install unsloth unsloth_zoo bitsandbytes
uv pip install -U transformers
# Install xformers from source with optimized build flags
uv pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers