# cd src/open-r1-multimodal

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_gui_grounding_lisa.py \
    --output_dir "out/lisa_train_GIoU" \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name NOT_USED \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 6 \
    --run_name Qwen2-VL-2B-GRPO-groud_lisa_train \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
