# Copyright (c) Alibaba, Inc. and its affiliates.
"""
On-Policy Distillation (OPD) Trainer

This trainer implements on-policy distillation where:
1. Student model generates rollouts (sampling trajectories)
2. For each generated token, compute both student and teacher log probabilities
3. Reward per token = teacher_logprob - student_logprob (negative reverse KL)
4. Update student model using REINFORCE: loss = -E[reward * log_prob]

Key difference from GKD:
- GKD: Uses JSD loss between teacher and student distributions (offline)
- OPD: Uses KL-based reward with policy gradient (online RL)
"""
import inspect
from collections import defaultdict, deque
from contextlib import nullcontext
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather_object
from transformers import PreTrainedModel
from trl import SFTTrainer as HFSFTTrainer

from swift.llm.template.template_inputs import TemplateInputs
from swift.utils import (
    JsonlWriter, get_logger, is_swanlab_available, is_wandb_available,
    remove_response, unwrap_model_for_generation
)
from ..mixin import SwiftMixin
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import (
    identity_data_collator, patch_profiling_context, patch_profiling_decorator,
    prepare_deepspeed
)

del HFSFTTrainer.__init__

logger = get_logger()
if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab


class OPDTrainer(RolloutTrainerMixin, SwiftMixin, HFSFTTrainer):
    """
    On-Policy Distillation Trainer

    Implements on-policy distillation using REINFORCE algorithm.
    Student model generates rollouts and receives rewards based on
    matching teacher's probability distribution.
    """

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        teacher_model = kwargs.pop('teacher_model')
        teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
        self.vllm_client = kwargs.pop('vllm_client', None)
        kwargs['data_collator'] = identity_data_collator

        super().__init__(model, None, *_args, **kwargs)
        args = kwargs['args']
        self.temperature = args.temperature
        self.generation_config = model.generation_config
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}

        # Store model kwarg keys for forward pass
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys() if not hasattr(model, 'get_base_model')
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Initialize teacher model with DeepSpeed support
        self.teacher_ds3_gather_for_generation = args.ds3_gather_for_generation
        self.is_teacher_ds3 = None
        if self.is_deepspeed_enabled:
            if teacher_deepspeed_config is not None:
                self.is_teacher_ds3 = teacher_deepspeed_config.get('zero_optimization', {}).get('stage') == 3
                if not self.is_teacher_ds3:
                    self.teacher_ds3_gather_for_generation = False
                self.teacher_model = prepare_deepspeed(
                    teacher_model, self.accelerator, deepspeed_config=teacher_deepspeed_config, training_args=args)
            else:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.teacher_model.eval()

        # Initialize rollout infrastructure for vLLM support
        if args.use_vllm:
            self.prepare_rollout()
            logger.info('vLLM engine initialized for OPD training')

        # Initialize logging
        self._prepare_logging()

    def _prepare_logging(self):
        """Initialize logging components for on-policy rollout tracking."""
        args = self.args
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = getattr(args, 'wandb_log_unique_prompts', False)
        import os
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))

        # Initialize logs deque for storing rollout data
        self._logs = {
            'prompt': deque(),
            'completion': deque(),
        }

    def _prepare_batch_inputs(self, inputs: list) -> Dict[str, torch.Tensor]:
        """Prepare batch inputs by encoding messages with response tokens"""
        template = self.template
        batch_encoded_inputs = []

        for data in inputs:
            if 'response_token_ids' in data and data['response_token_ids']:
                from .utils import replace_assistant_response_with_ids
                data['messages'] = replace_assistant_response_with_ids(data['messages'], data['response_token_ids'])

            encoded = template.encode(data, return_length=True)
            batch_encoded_inputs.append(encoded)

        from swift.llm import to_device
        batch_encoded = to_device(template.data_collator(batch_encoded_inputs), self.model.device)

        return batch_encoded

    @patch_profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute REINFORCE loss using teacher-student KL divergence as reward.

        For each token t:
            reward_t = teacher_logp_t - old_student_logp_t

        Policy gradient loss:
            loss = -mean(reward_t * new_student_logp_t)

        This encourages the student to assign high probability to tokens
        that the teacher also assigns high probability to.
        """
        # Prepare model inputs
        model_inputs = {k: v for k, v in inputs.items() if k not in {'labels'}}
        use_logits_to_keep = self.get_use_logits_to_keep(True)

        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            model_inputs['logits_to_keep'] = inputs['logits_to_keep']

        # Get student model outputs (current policy)
        outputs_student = model(**model_inputs, use_cache=False)
        logits_student = outputs_student.logits

        # Get teacher model outputs (for reward computation)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**model_inputs, use_cache=False)
            logits_teacher = outputs_teacher.logits

        # Prepare labels and mask for completion tokens
        shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
        mask = shifted_labels != -100

        # Extract logits for completion tokens only
        shifted_student_logits = logits_student[:, :-1, :][mask]
        shifted_teacher_logits = logits_teacher[:, :-1, :][mask]
        shifted_labels_masked = shifted_labels[mask]

        # Fix vocab size mismatch if needed (same as GKD)
        stu_dim = shifted_student_logits.shape[-1]
        tea_dim = shifted_teacher_logits.shape[-1]
        if stu_dim < tea_dim:
            shifted_student_logits = F.pad(shifted_student_logits, (0, tea_dim - stu_dim), 'constant', 0)
            shifted_student_logits[..., stu_dim:] = shifted_teacher_logits[..., stu_dim:]
        elif stu_dim > tea_dim:
            shifted_teacher_logits = F.pad(shifted_teacher_logits, (0, stu_dim - tea_dim), 'constant', 0)
            shifted_teacher_logits[..., tea_dim:] = shifted_student_logits[..., tea_dim:]

        # Compute log probabilities
        student_log_probs = F.log_softmax(shifted_student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(shifted_teacher_logits, dim=-1)

        # Get per-token log probabilities for actual tokens
        student_logps = torch.gather(student_log_probs, dim=1, index=shifted_labels_masked.unsqueeze(1)).squeeze(1)
        teacher_logps = torch.gather(teacher_log_probs, dim=1, index=shifted_labels_masked.unsqueeze(1)).squeeze(1)

        # Compute per-token rewards: teacher_logp - student_logp
        # This is equivalent to negative reverse KL: -KL(student || teacher)
        # Higher reward when student matches teacher better
        per_token_rewards = teacher_logps - student_logps.detach()

        # Normalize rewards (optional, helps with stability)
        rewards_mean = per_token_rewards.mean()
        rewards_std = per_token_rewards.std() + 1e-8
        normalized_rewards = (per_token_rewards - rewards_mean) / rewards_std

        # REINFORCE loss: -E[reward * log_prob]
        # We use normalized rewards as advantages
        loss = -(normalized_rewards.detach() * student_logps).mean()

        # Log metrics
        with torch.no_grad():
            mode = 'train' if self.model.training else 'eval'
            self._metrics[mode]['reward/mean'].append(per_token_rewards.mean().item())
            self._metrics[mode]['reward/std'].append(per_token_rewards.std().item())
            self._metrics[mode]['kl_divergence'].append((-per_token_rewards).mean().item())

        if return_outputs:
            return (loss, outputs_student)
        else:
            return loss

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        """Convert messages list to prompt text list using template"""
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    @patch_profiling_decorator
    def training_step(self, model: nn.Module, inputs: DataType,
                      num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        Perform a training step for on-policy distillation.

        This method:
        1. Generates rollouts using student model
        2. Computes rewards based on teacher-student KL divergence  
        3. Updates student model using REINFORCE
        """
        args = self.args

        # Generate completions and prepare inputs
        with patch_profiling_context(self, 'get_completions'):
            if args.use_vllm:
                processed_inputs = self._preprocess_inputs(inputs)
                generated_inputs = self._fast_infer(processed_inputs)
                if self.log_completions:
                    messages = [inp['messages'][:-1] for inp in generated_inputs]
                    completions = [deepcopy(inp['messages'][-1]['content']) for inp in generated_inputs]
                    valid_messages = gather_object(messages)
                    valid_completions = gather_object(completions)
                    self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
                    self._logs['completion'].extend(valid_completions)
                inputs = self._prepare_batch_inputs(generated_inputs)
            else:
                inputs = self._prepare_batch_inputs(inputs)
                with unwrap_model_for_generation(
                        model, self.accelerator,
                        gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                    unwrapped_model.eval()
                    # Generate on-policy outputs
                    from .gkd_trainer import GKDTrainer
                    new_input_ids, new_attention_mask, new_labels = GKDTrainer.generate_on_policy_outputs(
                        self, unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id)
                    unwrapped_model.train()
                inputs['input_ids'] = new_input_ids
                inputs['attention_mask'] = new_attention_mask
                inputs['labels'] = new_labels

        # Compute loss with forward context
        with self.template.forward_context(self.model, inputs):
            loss = HFSFTTrainer.training_step(self, model, inputs, num_items_in_batch)

        return loss

    def prediction_step(self, model, inputs, *args, **kwargs):
        """Prediction step for evaluation"""
        inputs = self._prepare_batch_inputs(inputs)
        with self.template.forward_context(self.model, inputs):
            return super().prediction_step(model, inputs, *args, **kwargs)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log method to include OPD-specific metrics and completion logging"""
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:
            super().log(logs)

        # Log OPD metrics
        if self.accelerator.is_main_process:
            mode = 'train' if self.model.training else 'eval'
            for metric_name, values in self._metrics[mode].items():
                if values:
                    avg_value = sum(values) / len(values)
                    logs[f'{mode}/{metric_name}'] = avg_value
            # Clear metrics after logging
            self._metrics[mode].clear()

        # Log completions table if we have data
        if self.accelerator.is_main_process and self.log_completions and len(self._logs['prompt']) > 0:
            seen_nums = len(self._logs['prompt'])
            table = {
                'step': [str(self.state.global_step)] * seen_nums,
                'prompt': list(self._logs['prompt'])[:seen_nums],
                'completion': list(self._logs['completion'])[:seen_nums],
            }

            # Write to jsonl
            self.jsonl_writer.append(table)

            self._logs['prompt'].clear()
            self._logs['completion'].clear()

            # Log to wandb if enabled
            report_to_wandb = self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None
            if report_to_wandb:
                wandb_table = table.copy()
                import pandas as pd
                df = pd.DataFrame(wandb_table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

            # Log to swanlab if enabled
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab.get_run(
            ) is not None
            if report_to_swanlab:
                headers = list(table.keys())
                rows = []
                for i in range(len(table['step'])):
                    row = [table[header][i] for header in headers]
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})
