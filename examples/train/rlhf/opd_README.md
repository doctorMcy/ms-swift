# 在线策略蒸馏 (On-Policy Distillation, OPD)

## 简介

在线策略蒸馏(On-Policy Distillation, OPD)是一种结合了强化学习和知识蒸馏的训练方法。与传统的离线蒸馏方法(如GKD)不同,OPD使用在线策略梯度算法来训练学生模型,使其能够更好地模仿教师模型的输出分布。

## 工作流程

OPD的训练流程可以分解为以下步骤:

### 1. 初始化教师客户端
- 教师模型可以是规模更大、能力更强的通用模型,也可以是经过专门训练的专家模型
- 教师模型只负责计算概率,不进行反向传播更新梯度
- 支持DeepSpeed Zero-3优化,可以处理大型教师模型

### 2. 学生模型采样轨迹
- 学生模型根据给定的提示(Prompt)自主生成完整的回答序列(Rollouts)
- 在生成过程中,记录每一步选择的token
- 支持使用vLLM加速推理,提高生成效率

### 3. 教师模型计算奖励
- 将学生模型生成的轨迹交给教师模型评估
- 教师模型对轨迹中的每个token计算对数概率
- 计算学生和教师的对数概率之差,得到每个token的分歧(Divergence)

### 4. 使用分歧作为奖励进行训练
- 使用负的逆向KL散度作为奖励信号:
  ```python
  reward_t = teacher_logp_t - student_logp_t
  ```
- 使用REINFORCE算法更新学生模型:
  ```python
  loss = -mean(reward_t * student_logp_t)
  ```
- 当学生模型的行为与教师模型一致时,KL散度为零,获得最高奖励
- 当学生模型的选择与教师模型差异很大时,KL散度变大,产生负奖励(惩罚)

## 核心实现

### Trainer类: `OPDTrainer`

位置: `swift/trainers/rlhf_trainer/opd_trainer.py`

关键方法:

1. **`compute_loss`**: 计算REINFORCE损失
   - 前向传播学生模型和教师模型
   - 计算每个token的奖励(teacher_logp - student_logp)
   - 归一化奖励并计算策略梯度损失

2. **`training_step`**: 执行单步训练
   - 生成学生模型的rollouts
   - 调用`compute_loss`计算损失
   - 更新学生模型参数

3. **`_prepare_batch_inputs`**: 准备批次输入
   - 编码消息和响应token
   - 准备模型前向传播所需的输入

## 与GKD的区别

| 特性 | GKD | OPD |
|------|-----|-----|
| 训练方式 | 离线蒸馏 | 在线强化学习 |
| 损失函数 | JSD散度 | REINFORCE策略梯度 |
| 数据生成 | 混合(教师/学生) | 纯学生生成 |
| 奖励信号 | 无 | 负的逆向KL散度 |
| 训练稳定性 | 更稳定 | 需要奖励归一化 |

## 使用示例

### 基础训练(无vLLM)

```bash
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
    --max_length 2048 \
    --max_completion_length 512 \
    --output_dir output/opd \
    --temperature 0.9 \
    --deepspeed zero2
```

### 使用vLLM加速

1. 启动vLLM rollout服务器:
```bash
CUDA_VISIBLE_DEVICES=7 \
swift rollout \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --vllm_max_model_len 4096
```

2. 运行训练:
```bash
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type opd \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset alpaca-zh#5000 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --deepspeed zero2 \
    --teacher_deepspeed zero3
```

## 主要参数

- `--rlhf_type opd`: 指定使用OPD算法
- `--teacher_model`: 教师模型路径
- `--teacher_deepspeed`: 教师模型的DeepSpeed配置
- `--temperature`: 生成时的温度参数(建议0.9)
- `--max_completion_length`: 最大生成长度
- `--use_vllm`: 是否使用vLLM加速
- `--log_completions`: 是否记录生成的完整回答

## 参考资料

- [On-Policy Distillation (Blog)](https://thinkingmachines.ai/blog/on-policy-distillation/)
- GKD论文: Generalized Knowledge Distillation for Language Models
- REINFORCE算法: Policy Gradient Methods for Reinforcement Learning

## 注意事项

1. **奖励归一化**: OPD使用奖励归一化来提高训练稳定性,这对于策略梯度方法至关重要

2. **教师模型大小**: 教师模型通常应该比学生模型大,以提供更好的知识指导

3. **温度参数**: 建议使用较高的温度(如0.9)来增加生成的多样性

4. **DeepSpeed支持**: 教师模型支持独立的DeepSpeed配置,可以使用Zero-3来节省显存

5. **vLLM加速**: 使用vLLM可以显著加速rollout生成,特别适合大批量训练

## 实现细节

### 奖励计算

```python
# 计算学生和教师的对数概率
student_logps = log_softmax(student_logits)[actual_tokens]
teacher_logps = log_softmax(teacher_logits)[actual_tokens]

# 奖励 = 负的逆向KL散度
reward = teacher_logps - student_logps.detach()

# 归一化奖励
normalized_reward = (reward - reward.mean()) / (reward.std() + 1e-8)
```

### REINFORCE损失

```python
# 策略梯度损失
loss = -mean(normalized_reward.detach() * student_logps)
```

这种实现确保:
- 学生模型倾向于选择教师也会选择的token(高奖励)
- 避免选择教师不太可能选择的token(低奖励/惩罚)
- 通过归一化提高训练稳定性
