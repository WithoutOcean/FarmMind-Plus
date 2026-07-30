#!/bin/bash

# =========================================================
# Farmland Cause 分布式训练脚本
# 使用 torchrun 在多个 GPU 上运行
# 使用方法: ./run_farmland_cause.sh
# =========================================================

conda init bash
source ~/.bashrc
export WANDB_DIR=Visual-RFT-origin/wandb
export WANDB_DATA_DIR=Visual-RFT-origin/wandb
eval "$(/opt/conda/bin/conda shell.bash hook)"
python Visual-RFT-origin/qwen3.py &
python Visual-RFT-origin/qwen3_copy.py &

EVAL_MODEL="Teacher model path"

GPUS="0,1"          
NNODES=1               
NPROC_PER_NODE=2  
noderank=0

export CUDA_VISIBLE_DEVICES=$GPUS

conda activate visualrf/

torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$noderank \
  Visual-RFT-origin/src/virft/src/open_r1/farmland_percept_reason.py


