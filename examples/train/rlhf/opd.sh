#!/bin/bash
# On-Policy Distillation (OPD)
# 在线策略蒸馏示例脚本
# 
# 工作流程:
# 1. 学生模型生成rollouts (轨迹采样)
# 2. 学生模型和教师模型分别计算生成token的对数概率
# 3. 使用负的逆向KL散度作为奖励: reward = teacher_logp - student_logp
# 4. 使用策略梯度方法更新学生模型

# 可选: 启动vLLM rollout服务器以加速推理
# CUDA_VISIBLE_DEVICES=7 \
# swift rollout \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --vllm_max_model_len 24192

# 使用内置引擎进行训练 (无需外部vLLM服务器)
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type opd \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset alpaca-zh#1000 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 2048 \
    --max_completion_length 512 \
    --output_dir output/opd \
    --warmup_ratio 0.1 \
    --temperature 0.9 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --save_only_model true


# 使用vLLM外部服务器进行加速推理
# NPROC_PER_NODE=2 \
# CUDA_VISIBLE_DEVICES=0,1 \
# swift rlhf \
#     --rlhf_type opd \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --teacher_model Qwen/Qwen2.5-7B-Instruct \
#     --train_type lora \
#     --dataset alpaca-zh#5000 \
#     --torch_dtype bfloat16 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --learning_rate 5e-5 \
#     --gradient_accumulation_steps 4 \
#     --save_steps 100 \
#     --save_total_limit 2 \
#     --logging_steps 10 \
#     --max_length 2048 \
#     --max_completion_length 512 \
#     --output_dir output/opd_vllm \
#     --warmup_ratio 0.1 \
#     --temperature 0.9 \
#     --dataloader_num_workers 4 \
#     --dataset_num_proc 4 \
#     --deepspeed zero2 \
#     --teacher_deepspeed zero3 \
#     --use_vllm true \
#     --vllm_mode server \
#     --vllm_server_host 127.0.0.1 \
#     --vllm_server_port 8000 \
#     --save_only_model true
