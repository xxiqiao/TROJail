from verl.trainer.ppo.core_algos import *
import pdb
from collections import defaultdict
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

# supported by Kangrui Wang
def compute_bi_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        reward_mask = token_level_rewards.bool()
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()

        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions=reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]

                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]

                else:
                    # Last valid position
                    nextvalue = 0.0

                # Calculate delta using the next valid token
                delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]

                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam

            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]

            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)

    return advantages, returns

def compute_grpo_process_advantage_bak(
    token_level_process_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_ids: np.ndarray = None,
):
    with torch.no_grad():
        bsz, seqlen = token_level_process_rewards.shape
        device = token_level_process_rewards.device



        process_rewards = []
        trajectory_turn_info = []  # Store (trajectory_idx, turn_idx, original_position) for mapping back

        for i in range(bsz):
            mask = response_mask[i]  # shape: (seq_len,)
            scores = token_level_process_rewards[i]  # shape: (seq_len,)

            # Find the indices where response_mask is True
            response_indices = torch.where(mask)[0]

            # Group consecutive indices to identify turns
            if len(response_indices) > 0:
                segments = split_trun_from_mask(mask)

                # Extract rewards at the end of each turn (last position of each segment)
                turn_rewards = []
                for j, seg in enumerate(segments):
                    if len(seg) > 0:
                        end_idx = seg[-1].item()
                        turn_rewards.append(scores[end_idx])
                        trajectory_turn_info.append((i, j, end_idx))

                if len(turn_rewards) > 0:
                    process_rewards.extend(turn_rewards)
                else:
                    raise ValueError(f"No valid turns found for trajectory {i}")
            else:
                # If no response tokens found, raise error
                raise ValueError(f"No response tokens found for trajectory {i}")

        # Convert to tensor for normalization
        all_process_rewards = torch.stack(process_rewards)



        id2process_rewards = defaultdict(list)
        for i, (traj_idx, turn_idx, pos_idx) in enumerate(trajectory_turn_info):
            id2process_rewards[index[traj_idx]].append(all_process_rewards[i])


        id2mean = {}
        id2std = {}
        for idx, values in id2process_rewards.items():
            if len(values) == 1:
                id2mean[idx] = torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                stacked = torch.stack(values)
                id2mean[idx] = torch.mean(stacked)
                id2std[idx] = torch.std(stacked)


        normalized_rewards = all_process_rewards.clone()
        for i, (traj_idx, turn_idx, pos_idx) in enumerate(trajectory_turn_info):
            m = id2mean[index[traj_idx]]
            s = id2std[index[traj_idx]]
            if norm_adv_by_std_in_grpo:
                normalized_rewards[i] = (all_process_rewards[i] - m) / (s + epsilon)
            else:
                normalized_rewards[i] = all_process_rewards[i] - m


        normalized_process_rm_scores = token_level_process_rewards.clone()
        for reward_idx, (traj_idx, turn_idx, pos_idx) in enumerate(trajectory_turn_info):
            normalized_process_rm_scores[traj_idx, pos_idx] = normalized_rewards[reward_idx]


        discounted = torch.zeros_like(normalized_process_rm_scores)
        for i in range(bsz):
            mask = response_mask[i]
            rewards = normalized_process_rm_scores[i]
            segments = split_trun_from_mask(mask)

            turn_rewards = []
            for seg in segments:
                if len(seg) == 0:
                    continue
                final_idx = seg[-1]
                turn_rewards.append(rewards[final_idx])

            acc = 0.0
            turn_returns = []
            for r in reversed(turn_rewards):
                acc = r + gamma * acc
                turn_returns.append(acc)
            turn_returns = list(reversed(turn_returns))

            for seg, R in zip(segments, turn_returns):
                discounted[i, seg] = R

        advantages = discounted * response_mask

    return advantages, discounted * response_mask

def compute_grpo_process_advantage(
    token_level_process_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_ids: np.ndarray = None,
):
    with torch.no_grad():
        bsz, seqlen = token_level_process_rewards.shape
        # normalized_process_rm_scores = token_level_process_rewards / (1.0 + epsilon)
        normalized_process_rm_scores = token_level_process_rewards

        discounted = torch.zeros_like(normalized_process_rm_scores)
        for i in range(bsz):
            mask = response_mask[i]
            rewards = normalized_process_rm_scores[i]
            segments = split_trun_from_mask(mask)

            turn_rewards = []
            for seg in segments:
                if len(seg) == 0:
                    continue
                final_idx = seg[-1]
                turn_rewards.append(rewards[final_idx])

            acc = 0.0
            turn_returns = []
            for r in reversed(turn_rewards):
                acc = r + gamma * acc
                turn_returns.append(acc)
            turn_returns = list(reversed(turn_returns))

            for seg, R in zip(segments, turn_returns):
                discounted[i, seg] = R

        advantages = discounted * response_mask

    return advantages, discounted * response_mask

def compute_grpo_token_level_process_advantage(
    token_level_process_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_ids: np.ndarray = None,
):
    """
    Compute token-level discounted returns and advantages for GRPO.
    Different from turn-level: here we use every response token's reward,
    and accumulate across the whole trajectory without resetting at turn boundaries.
    """
    with torch.no_grad():
        bsz, seqlen = token_level_process_rewards.shape
        rewards = token_level_process_rewards  # already normalized outside

        discounted = torch.zeros_like(rewards)

        for i in range(bsz):
            mask = response_mask[i]  # (seqlen,)
            r = rewards[i]

            acc = 0.0
            # scan backwards across the whole trajectory
            for t in reversed(range(seqlen)):
                if mask[t]:
                    acc = r[t] + gamma * acc
                    discounted[i, t] = acc


        advantages = discounted * response_mask

    return advantages, discounted * response_mask



# def compute_grpo_process_advantage(
#     token_level_process_rewards: torch.Tensor,
#     response_mask: torch.Tensor,
#     index: np.ndarray,
#     gamma: float = 1.0,
#     epsilon: float = 1e-6,
#     norm_adv_by_std_in_grpo: bool = True,
#     group_ids: np.ndarray = None,
# ):
#     with torch.no_grad():
#         bsz, seqlen = token_level_process_rewards.shape
#         device = token_level_process_rewards.device
#         flat_rewards = torch.zeros_like(token_level_process_rewards)


#         for i in range(bsz):
#             mask = response_mask[i]
#             rewards = token_level_process_rewards[i]
#             segments = split_trun_from_mask(mask)

#             for seg in segments:
#                 if len(seg) == 0:
#                     continue
#                 final_idx = seg[-1]
#                 reward = rewards[final_idx]
#                 flat_rewards[i, seg] = reward


#         sample_flat = flat_rewards.sum(dim=-1)  # (bsz,)
#         id2score = defaultdict(list)
#         for i in range(bsz):
#             id2score[index[i]].append(sample_flat[i])

#         id2mean = {}
#         id2std = {}
#         for idx, values in id2score.items():
#             if len(values) == 1:
#                 id2mean[idx] = torch.tensor(0.0)
#                 id2std[idx] = torch.tensor(1.0)
#             else:
#                 stacked = torch.stack(values)
#                 id2mean[idx] = torch.mean(stacked)
#                 id2std[idx] = torch.std(stacked)


#         normalized_rewards = torch.zeros_like(flat_rewards)
#         for i in range(bsz):
#             m = id2mean[index[i]]
#             s = id2std[index[i]]
#             if norm_adv_by_std_in_grpo:
#                 normalized_rewards[i] = (flat_rewards[i] - m) / (s + epsilon)
#             else:
#                 normalized_rewards[i] = flat_rewards[i] - m


#         discounted = torch.zeros_like(normalized_rewards)
#         for i in range(bsz):
#             mask = response_mask[i]
#             rewards = normalized_rewards[i]
#             segments = split_trun_from_mask(mask)

#             turn_rewards = []
#             for seg in segments:
#                 if len(seg) == 0:
#                     continue
#                 final_idx = seg[-1]
#                 turn_rewards.append(rewards[final_idx])

#             acc = 0.0
#             turn_returns = []
#             for r in reversed(turn_rewards):
#                 acc = r + gamma * acc
#                 turn_returns.append(acc)
#             turn_returns = list(reversed(turn_returns))

#             for seg, R in zip(segments, turn_returns):
#                 discounted[i, seg] = R

#         advantages = discounted * response_mask

#     return advantages, discounted * response_mask

# set up unittest
if __name__ == "__main__":
    token_level_rewards = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    loss_mask = torch.ones(1, 10)
    advantages, returns = compute_bi_level_gae_advantage_return(token_level_rewards, values, loss_mask, 1, 1, 0.95)
    print(advantages)
    print(returns)