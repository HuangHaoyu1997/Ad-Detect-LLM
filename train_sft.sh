CUDA_VISIBLE_DEVICES=0 \
swift sft  \
    --model /mnt/e/LLM/MiniCPM4-0.5B \
    --train_type lora \
    --dataset ./sft_dataset/train.json \
    --val_dataset ./sft_dataset/test.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 3e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 20 \
    --save_steps 20 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir ./output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4

# r=16对大部分任务来说是个不错的起点。实际上，参数关系遵循`α/r`的缩放因子，其中α通常设为r的2倍，这个"alpha = 2 × rank"的启发式规则已经被HuggingFace PEFT库广泛采用。
# /mnt/e/LLM/MiniCPM4-0.5B
# Qwen/Qwen3-0.6B
