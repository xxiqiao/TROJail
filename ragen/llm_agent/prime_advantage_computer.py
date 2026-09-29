"""
PRIME Advantage Computer for Multi-turn Jailbreak
Computes combined GRPO advantages using both outcome rewards and turn-level process rewards.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from verl import DataProto
from verl.utils.torch_functional import masked_whiten
from collections import defaultdict
import pdb


def compute_prime_grpo_advantage(
    data: DataProto,
    outcome_rewards: torch.Tensor,
    process_rewards: torch.Tensor,
    config,
    group_ids: Optional[np.ndarray] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute combined GRPO advantage using both outcome and process rewards.
    
    Args:
        data: DataProto containing dialogue data
        outcome_rewards: torch.Tensor of outcome rewards (final scores from judge LLM)
        process_rewards: torch.Tensor of turn-level process rewards
        config: Configuration object
        group_ids: Group IDs for GRPO grouping
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Combined advantages and returns
    """
    # Get PRIME hyperparameters
    lambda_1 = config.algorithm.get("prime_lambda_1", 0.1)  # Weight for process rewards
    gamma = config.algorithm.get("prime_gamma", 0.95)  # Discount factor for process rewards
    norm_adv_by_std = config.algorithm.get("norm_adv_by_std_in_grpo", True)
    
    # Use provided group_ids or extract from data
    if group_ids is None:
        group_ids = data.non_tensor_batch.get("group_ids", np.arange(len(outcome_rewards)))
    
    # Compute GRPO advantages for outcome rewards
    outcome_advantages = _compute_grpo_advantage(
        outcome_rewards, group_ids, norm_adv_by_std, "outcome"
    )
    
    # Compute GRPO advantages for process rewards
    process_advantages = _compute_grpo_advantage(
        process_rewards, group_ids, norm_adv_by_std, "process"
    )
    
    # Apply discounting to process advantages
    discounted_process_advantages = _apply_turn_discounting(
        process_advantages, data, gamma
    )
    
    # Combine advantages according to the formula
    combined_advantages = outcome_advantages + lambda_1 * discounted_process_advantages
    
    # Compute returns (same as advantages for GRPO)
    returns = combined_advantages.clone()
    
    return combined_advantages, returns


def _compute_grpo_advantage(
    rewards: torch.Tensor,
    group_ids: np.ndarray,
    norm_adv_by_std: bool,
    reward_type: str
) -> torch.Tensor:
    """
    Compute GRPO advantage for given rewards.
    
    Args:
        rewards: torch.Tensor of rewards
        group_ids: Group IDs for GRPO grouping
        norm_adv_by_std: Whether to normalize by standard deviation
        reward_type: Type of reward ("outcome" or "process")
        
    Returns:
        torch.Tensor: GRPO advantages
    """
    # Group rewards by group_id
    id2rewards = defaultdict(list)
    for i, group_id in enumerate(group_ids):
        id2rewards[group_id].append(rewards[i])
    
    # Compute mean and std for each group
    id2mean = {}
    id2std = {}
    for group_id, group_rewards in id2rewards.items():
        if len(group_rewards) == 1:
            id2mean[group_id] = torch.tensor(0.0, device=rewards.device)
            id2std[group_id] = torch.tensor(1.0, device=rewards.device)
        else:
            group_tensor = torch.stack(group_rewards)
            id2mean[group_id] = group_tensor.mean()
            id2std[group_id] = group_tensor.std()
    
    # Compute normalized advantages
    advantages = torch.zeros_like(rewards)
    for i, group_id in enumerate(group_ids):
        if norm_adv_by_std:
            advantages[i] = (rewards[i] - id2mean[group_id]) / (id2std[group_id] + 1e-6)
        else:
            advantages[i] = rewards[i] - id2mean[group_id]
    
    return advantages


def _apply_turn_discounting(
    process_advantages: torch.Tensor,
    data: DataProto,
    gamma: float
) -> torch.Tensor:
    """
    Apply turn-level discounting to process advantages.
    
    Args:
        process_advantages: Turn-level process advantages
        data: DataProto containing dialogue data
        gamma: Discount factor
        
    Returns:
        torch.Tensor: Discounted process advantages
    """
    # Get turn information from dialogue history
    turn_lengths = _extract_turn_lengths(data)
    
    discounted_advantages = torch.zeros_like(process_advantages)
    
    for i, num_turns in enumerate(turn_lengths):
        if i < process_advantages.shape[0]:
            # Apply discounting: later turns get higher weight
            for turn in range(num_turns):
                discount_factor = gamma ** (num_turns - turn - 1)
                discounted_advantages[i] += process_advantages[i] * discount_factor
    
    return discounted_advantages


def _extract_turn_lengths(data: DataProto) -> List[int]:
    """
    Extract the number of turns for each dialogue.
    
    Args:
        data: DataProto containing dialogue data
        
    Returns:
        List[int]: Number of turns for each dialogue
    """
    turn_lengths = []
    
    # Extract from dialogue history if available
    if "dialogue_history" in data.non_tensor_batch:
        for dialogue in data.non_tensor_batch["dialogue_history"]:
            # Count turns (each turn has an LLM response)
            num_turns = sum(1 for entry in dialogue if "llm_response" in entry)
            turn_lengths.append(num_turns)
    else:
        # Fallback: estimate from response tokens
        responses = data.batch["responses"]
        for i in range(responses.shape[0]):
            # Count EOS tokens or turn markers
            response_tokens = responses[i]
            num_turns = 1  # At least one turn
            for token in response_tokens:
                if token in [2, 151645]:  # EOS token IDs (adapt based on your tokenizer)
                    num_turns += 1
            turn_lengths.append(num_turns)
    
    return turn_lengths


def extract_outcome_rewards_from_dialogue(data: DataProto) -> torch.Tensor:
    """
    Extract outcome rewards from dialogue history.
    
    Args:
        data: DataProto containing dialogue data
        
    Returns:
        torch.Tensor: Outcome rewards for each dialogue
    """
    outcome_rewards = []
    
    if "dialogue_history" in data.non_tensor_batch:
        for dialogue in data.non_tensor_batch["dialogue_history"]:
            # Get the final score from the last turn
            final_score = 0.0
            for entry in dialogue:
                if "score" in entry:
                    final_score = entry["score"]
            outcome_rewards.append(final_score)
    else:
        # Fallback: use existing reward scores if available
        if "rm_scores" in data.batch:
            outcome_rewards = data.batch["rm_scores"].sum(dim=-1)
        else:
            # Default to zero rewards
            outcome_rewards = torch.zeros(data.batch["responses"].shape[0], device=data.batch["responses"].device)
    
    if isinstance(outcome_rewards, list):
        outcome_rewards = torch.tensor(outcome_rewards, device=data.batch["responses"].device)
    
    return outcome_rewards


def compute_turn_level_process_rewards(
    data: DataProto,
    prime_model=None,
    ref_model=None,
    beta: float = 0.05
) -> torch.Tensor:
    """
    Compute turn-level process rewards using implicit PRM.
    
    Args:
        data: DataProto containing dialogue data
        prime_model: Implicit PRM model
        ref_model: Reference model
        beta: Temperature for KL divergence
        
    Returns:
        torch.Tensor: Turn-level process rewards
    """
    if prime_model is None or ref_model is None:
        # If no PRM is available, return zero rewards
        batch_size = data.batch["responses"].shape[0]
        return torch.zeros(batch_size, device=data.batch["responses"].device)
    
    # Extract turn-level responses
    turn_responses = _extract_turn_responses(data)
    
    # Compute process rewards for each dialogue
    process_rewards = []
    for i, responses in enumerate(turn_responses):
        if len(responses) == 0:
            process_rewards.append(torch.tensor(0.0, device=data.batch["responses"].device))
            continue
        
        # Compute average process reward across turns
        turn_rewards = []
        for response in responses:
            # Compute KL divergence between PRM and reference model
            with torch.no_grad():
                # This is a simplified computation - implement based on your model interface
                kl_div = _compute_kl_divergence(prime_model, ref_model, response, beta)
                turn_rewards.append(kl_div)
        
        if turn_rewards:
            avg_reward = torch.stack(turn_rewards).mean()
        else:
            avg_reward = torch.tensor(0.0, device=data.batch["responses"].device)
        
        process_rewards.append(avg_reward)
    
    return torch.stack(process_rewards) if process_rewards else torch.zeros(1, device=data.batch["responses"].device)


def _extract_turn_responses(data: DataProto) -> List[List[str]]:
    """
    Extract turn-level responses from dialogue history.
    
    Args:
        data: DataProto containing dialogue data
        
    Returns:
        List[List[str]]: List of turn responses for each dialogue
    """
    turn_responses = []
    
    if "dialogue_history" in data.non_tensor_batch:
        for dialogue in data.non_tensor_batch["dialogue_history"]:
            turns = []
            for entry in dialogue:
                if "llm_response" in entry:
                    turns.append(entry["llm_response"])
            turn_responses.append(turns)
    else:
        # Fallback: extract from response tokens
        responses = data.batch["responses"]
        for i in range(responses.shape[0]):
            # This would need to be adapted based on your tokenization
            # For now, return empty list
            turn_responses.append([])
    
    return turn_responses


def _compute_kl_divergence(prime_model, ref_model, response: str, beta: float) -> torch.Tensor:
    """
    Compute KL divergence between PRM and reference model for a response.
    
    Args:
        prime_model: Implicit PRM model
        ref_model: Reference model
        response: Response string
        beta: Temperature parameter
        
    Returns:
        torch.Tensor: KL divergence
    """
    # This is a placeholder implementation
    # You'll need to implement based on your model interface
    # For now, return a random value for demonstration
    return torch.tensor(0.0, device=next(prime_model.parameters()).device)


def convert_to_token_level_rewards(
    turn_advantages: torch.Tensor,
    data: DataProto
) -> torch.Tensor:
    """
    Convert turn-level advantages to token-level rewards.
    
    Args:
        turn_advantages: Turn-level advantages
        data: DataProto containing dialogue data
        
    Returns:
        torch.Tensor: Token-level rewards
    """
    # Get response mask
    response_mask = data.batch.get("response_mask", data.batch.get("loss_mask"))
    
    # Create token-level rewards tensor
    token_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    
    # For simplicity, distribute the advantage evenly across all tokens
    # In practice, you'd want to distribute based on turn boundaries
    for i, advantage in enumerate(turn_advantages):
        if i < token_rewards.shape[0]:
            # Count valid tokens
            valid_tokens = response_mask[i].sum()
            if valid_tokens > 0:
                token_rewards[i] = advantage / valid_tokens * response_mask[i]
    
    return token_rewards