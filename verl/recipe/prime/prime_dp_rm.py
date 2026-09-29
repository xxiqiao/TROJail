# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import itertools
import logging
import os
from typing import Tuple

import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_, offload_fsdp_model_to_cpu
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

from .prime_core_algos import compute_ce_dpo_loss_rm, compute_detach_dpo_loss_rm

from peft import PeftModel

import pdb

__all__ = ["DataParallelPRIMERewardModel"]


def split_trun_from_mask(mask):
    indices = torch.where(mask)[0]
    # splits = torch.where(indices[1:] != indices[:-1] + 1)[0] + 1
    # segments = torch.split(indices, splits.tolist())
    if len(indices) == 0:
        return []

    if len(indices) > 1:
        splits = torch.where(indices[1:] != indices[:-1] + 1)[0] + 1

        split_sizes = []
        if len(splits) > 0:
            split_sizes.append(splits[0].item())
            for j in range(len(splits) - 1):
                split_sizes.append((splits[j+1] - splits[j]).item())
            split_sizes.append(len(indices) - splits[-1].item())
        else:
            split_sizes = [len(indices)]
    else:
        split_sizes = [1]

    segments = torch.split(indices, split_sizes)
    return segments


class DataParallelPRIMERewardModel:
    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Reward model use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

    def _forward_micro_batch_bak(self, micro_batch, prompt_length):
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]

        num_actions = micro_batch["input_ids"].shape[-1] - prompt_length
        max_positions = micro_batch["attention_mask"][:, prompt_length:].sum(-1)

        if self.use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)
            position_ids_rmpad = position_ids_rmpad - 1 # Added to solve the problem of https://github.com/volcengine/verl/issues/1270

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            rm_output_logits = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False).logits.squeeze(0)  # copied. I don't really know why there is a squeeze
            rm_log_labels = verl_F.logprobs_from_logits(logits=rm_output_logits, labels=input_ids_rmpad_rolled)
            if self.ulysses_sequence_parallel_size > 1:
                rm_log_labels = gather_outpus_and_unpad(rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
            rm_log_labels = pad_input(hidden_states=rm_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -num_actions - 1 : -1]

        else:
            rm_output_logits = self.reward_module(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                position_ids=micro_batch["position_ids"],
                use_cache=False,
            ).logits
            rm_log_prob = torch.nn.functional.log_softmax(rm_output_logits[:, :-1, :], dim=-1)  # (batch_size, seq_length, vocab_size)
            rm_log_labels = rm_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)  # (batch, seq_length)

        if self.ref_module is not None:
            # do not have to pad again
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.ulysses_sequence_parallel_size > 1 and self.use_remove_padding:
                    ref_output_logits = self.ref_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False).logits.squeeze(0)
                    ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits, labels=input_ids_rmpad_rolled)
                    ref_log_labels = gather_outpus_and_unpad(ref_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    ref_log_labels = pad_input(hidden_states=ref_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -num_actions - 1 : -1]
                else:
                    ref_output_logits = self.ref_module(
                        input_ids=micro_batch["input_ids"],
                        attention_mask=micro_batch["attention_mask"],
                        position_ids=micro_batch["position_ids"],
                        use_cache=False,
                    ).logits
                    ref_log_prob = torch.nn.functional.log_softmax(ref_output_logits[:, :-1, :], dim=-1)  # (batch_size, seq_length, vocab_size)
                    ref_log_labels = ref_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)  # (batch, seq_length)
        else:
            ref_log_labels = micro_batch["old_log_probs"]

        ref_log_labels.to(rm_log_labels.dtype)
        q = rm_log_labels[:, -num_actions:] - ref_log_labels[:, -num_actions:]  # this is actually diff of q

        # trim unnecessary logprobs here
        for i in range(micro_batch["input_ids"].shape[0]):
            q[i, max_positions[i] :] = 0

        # reward computation does not need gradient. only q needs
        with torch.no_grad():
            # generalized estimation of r should go before the reward filling. r means process reward for policy model, or the advantage of reward model.
            lam = self.config.get("lambda", 0.0)
            beta = self.config.model.get("beta_train", 0.05)
            if lam == 0.0:
                r = q * beta
            else:
                # reward coefficient takes no effect here
                acc = micro_batch["acc"]
                q_ = q * beta
                r = torch.zeros_like(q)
                lastgaelam = 0
                # change the last token and mask out all paddings to make this process easier if we rely on outcome reward to calculate V
                for i in range(q.shape[0]):
                    if self.config.prime_use_gt:
                        q_[i, max_positions[i] - 1] = acc[i] - q_[i, : max_positions[i] - 1].sum()
                    q_[i, max_positions[i] :] = 0

                for t in reversed(range(num_actions)):
                    delta = q_[:, t]
                    lastgaelam = delta + lam * lastgaelam
                    r[:, t] = lastgaelam

            token_level_score = torch.zeros_like(q)

            if self.config.prime_granularity == "token":
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, : max_positions[i] - 1] = r[i, : max_positions[i] - 1]
            elif self.config.prime_granularity == "whole":
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, max_positions[i] - 1] = r[i, : max_positions[i]]
            else:
                raise NotImplementedError

        return token_level_score, q

    def _forward_micro_batch(self, micro_batch, is_offload_param: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            response_mask = micro_batch["response_mask"]
            position_ids = micro_batch["position_ids"]
            valid_lens = micro_batch["attention_mask"].sum(-1)
            empty_assistant_mask = micro_batch["empty_assistant_mask"]
            # pdb.set_trace()
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)
                position_ids_rmpad = position_ids_rmpad - 1 # Added to solve the problem of https://github.com/volcengine/verl/issues/1270
                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    # pdb.set_trace()
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                # pdb.set_trace()
                output = self.reward_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                rm_output_logits = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                # rm_output_logits.div_(temperature)
                rm_log_labels = logprobs_from_logits(logits=rm_output_logits, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    # gather and unpad for the ulysses sp
                    rm_log_labels = gather_outpus_and_unpad(rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                rm_log_labels = pad_input(hidden_states=rm_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                rm_log_labels = rm_log_labels.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.reward_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                # logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                rm_log_labels = logprobs_from_logits(logits, micro_batch["responses"])

        if self.ref_module is not None:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.ulysses_sequence_parallel_size > 1 and self.use_remove_padding:
                    ref_output_logits = self.ref_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False).logits.squeeze(0)
                    ref_log_labels = logprobs_from_logits(logits=ref_output_logits, labels=input_ids_rmpad_rolled)
                    ref_log_labels = gather_outpus_and_unpad(ref_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    ref_log_labels = pad_input(hidden_states=ref_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -response_length - 1 : -1]
                else:
                    ref_output = self.ref_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                    )
                    ref_output_logits = ref_output.logits
                    ref_output_logits = ref_output_logits[:, -response_length - 1 : -1, :]
                    ref_log_labels = logprobs_from_logits(ref_output_logits, micro_batch["responses"])

        ref_log_labels.to(rm_log_labels.dtype)
        q = rm_log_labels[:, -response_length:] - ref_log_labels[:, -response_length:]
        # q = rm_log_labels - ref_log_labels
        # trim unnecessary logprobs here
        # pdb.set_trace()
        for i in range(micro_batch["input_ids"].shape[0]):
            q[i, : -valid_lens[i]] = 0

        # check rm_log_labels.shape and ref_log_labels.shape, response_length, valid_lens
        # check lam

        # pdb.set_trace()

        with torch.no_grad():
            # generalized estimation of r should go before the reward filling. r means process reward for policy model, or the advantage of reward model.
            lam = self.config.get("lambda", 0.0)
            beta = self.config.model.get("beta_train", 0.05)
            if lam == 0.0:
                r = q * beta
            else:
                raise NotImplementedError

            token_level_score = torch.zeros_like(q)


            if self.config.prime_granularity == "token":
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, -valid_lens[i]:] = r[i, -valid_lens[i]:]
            elif self.config.prime_granularity == "whole":
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, -1] = r[i, -valid_lens[i]:].sum()
            elif self.config.prime_granularity == "turn":
                filter_empty_response = self.config.get("filter_empty_response", False)
                for i, mask in enumerate(response_mask):  # mask shape: (seq_len,)
                    empty_assistant_mask_i = empty_assistant_mask[i]
                    segments = split_trun_from_mask(mask)

                    for j, seg in enumerate(segments):
                        if len(seg) == 0:
                            continue
                        # pdb.set_trace()
                        start, end = seg[0].item(), seg[-1].item()
                        if filter_empty_response:
                            if not empty_assistant_mask_i[j]:
                                q_sum = r[i, start-1 : end].sum()
                                # token_level_score[i][end-1] = q_sum
                                token_level_score[i][end] = q_sum
                        else:
                            q_sum = r[i, start-1 : end].sum()
                            token_level_score[i][end] = q_sum


            else:
                raise NotImplementedError

            # if is_offload_param:
            #     offload_fsdp_model_to_cpu(self.ref_module)

            return token_level_score, q

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        elif isinstance(self.reward_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip)
        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.reward_optimizer.zero_grad()
        else:
            self.reward_optimizer.step()
        return grad_norm

    def prime_norm(self, token_level_scores):
        if self.config.prime_norm == "batch_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def compute_rm_score(self, data: DataProto):
        self.reward_module.eval()
        self.ref_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "response_mask", "acc"]
        batch = data.select(batch_keys=select_keys).batch

        messages_list = data.non_tensor_batch["messages_list"]

        empty_assistant_turns = []
        max_turns = 0
        for messages in messages_list:
            flags = [m.get("content", "") == "" for m in messages if m.get("role") == "assistant"]
            empty_assistant_turns.append(flags)
            max_turns = max(max_turns, len(flags))

        padded_flags = []
        for flags in empty_assistant_turns:
            padded = flags + [False]*(max_turns - len(flags))
            padded_flags.append(padded)

        empty_mask_tensor = torch.tensor(padded_flags, dtype=torch.bool)


        batch["empty_assistant_mask"] = empty_mask_tensor

        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        # prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]
        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst = []
        q_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                # rm_score, q = self._forward_micro_batch(micro_batch, prompt_length)
                rm_score, q = self._forward_micro_batch(micro_batch)
            rm_scores_lst.append(rm_score)
            q_lst.append(q)
        rm_scores = torch.concat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]

        return (
            rm_scores,
            q.detach(),
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            },
        )

    def update_rm(self, data: DataProto, is_offload_param: bool = False):
        # make sure we are in training mode
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        # select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "response_mask", "acc"]

        for key in ["Q_bc", "acc_bc"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347

        messages_list = data.non_tensor_batch["messages_list"]

        empty_assistant_turns = []
        max_turns = 0
        for messages in messages_list:
            flags = [m.get("content", "") == "" for m in messages if m.get("role") == "assistant"]
            empty_assistant_turns.append(flags)
            max_turns = max(max_turns, len(flags))

        padded_flags = []
        for flags in empty_assistant_turns:
            padded = flags + [False]*(max_turns - len(flags))
            padded_flags.append(padded)

        empty_mask_tensor = torch.tensor(padded_flags, dtype=torch.bool)


        batch["empty_assistant_mask"] = empty_mask_tensor

        dataloader = batch.split(self.config.mini_batch_size)
        # print("DEBUG: len(dataloader)\n", len(dataloader))

        rm_scores_lst = []
        q_lst = []

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            # print("DEBUG: mini_batch\n", mini_batch)
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            # print("DEBUG: len(micro_batches)\n", len(micro_batches))
            self.reward_optimizer.zero_grad()

            for data in micro_batches:
                # print("DEBUG: micro_batch\n", data)
                # pdb.set_trace()
                # data = data.cuda()
                if isinstance(data, DataProto):
                    data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                else:
                    data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                # attention_mask = data["attention_mask"]
                acc = data["acc"]
                response_mask = data["response_mask"]

                # prompt_ids = data["prompts"]
                # prompt_length = prompt_ids.shape[-1]

                # response_mask = attention_mask[:, prompt_length:]

                # pdb.set_trace()
                rm_score, q = self._forward_micro_batch(data, is_offload_param)

                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

                if self.config.model.loss_type == "ce":

                    dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                else:
                    raise NotImplementedError

                data = {"reward_model/dpo_loss": dpo_loss.detach().item()}

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = dpo_loss * (len(data) / self.config.mini_batch_size)
                else:
                    loss = dpo_loss / self.gradient_accumulation

                loss.backward()

                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {"reward_model/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.reward_optimizer.zero_grad()

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        metrics.update(
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            }
        )

        return rm_scores, metrics
