cd src/virft
# pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3 -i https://mirrors.zju.edu.cn/pypi/web/simple
pip install tensorboardx -i https://mirrors.zju.edu.cn/pypi/web/simple
pip install qwen_vl_utils torchvision -i https://mirrors.zju.edu.cn/pypi/web/simple
# pip install flash-attn --no-build-isolation -i https://mirrors.zju.edu.cn/pypi/web/simple

# vLLM support 
# pip install vllm==0.7.2

# fix transformers version
#pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
