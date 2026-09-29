"""
PRIME (Process Reinforcement through IMplicit rEwards) Reward Manager for Multi-turn Jailbreak
Adapted from the original PRIME implementation to work with multi-turn conversations and GRPO.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from verl import DataProto
from verl.utils.torch_functional import masked_whiten
from collections import defaultdict
import pdb


class JailbreakPrimeRewardManager:
    """
    PRIME Reward Manager for multi-turn jailbreak scenarios.
    Provides turn-level implicit process rewards combined with outcome rewards using GRPO.
    """
    
    def __init__(self, config, tokenizer, prime_model=None, ref_model=None):
        self.config = config
        self.tokenizer = tokenizer
        self.prime_model = prime_model  # Implicit PRM for process rewards
        self.ref_model = ref_model      # Reference model for KL divergence
        self.lambda_1 = config.algorithm.get("prime_lambda_1", 0.1)  # Weight for process rewards
        self.beta = config.algorithm.get("prime_beta", 0.05)  # Temperature for KL divergence
        self.gamma = config.algorithm.get("prime_gamma", 0.95)  # Discount factor for process rewards
        
    def compute_turn_level_process_rewards(self, data: DataProto) -> torch.Tensor:
        """
        Compute turn-level process rewards using the implicit PRM.
        
        Args:
            data: DataProto containing dialogue history and responses
            
        Returns:
            torch.Tensor: Turn-level process rewards for each turn in each dialogue
        """
        if self.prime_model is None or self.ref_model is None:
            # If no PRM is available, return zero rewards
            batch_size = data.batch["responses"].shape[0]
            max_turns = data.batch["responses"].shape[1] if len(data.batch["responses"].shape) > 1 else 1
            return torch.zeros(batch_size, max_turns, device=data.batch["responses"].device)
        
        # Extract turn-level responses from the dialogue history
        turn_responses = self._extract_turn_responses(data)
        
        # Compute KL divergence between PRM and reference model for each turn
        process_rewards = []
        for i, responses in enumerate(turn_responses):
            if len(responses) == 0:
                process_rewards.append(torch.zeros(1, device=data.batch["responses"].device))
                continue
                
            # Compute log probabilities for each turn response
            with torch.no_grad():
                # This is a simplified version - in practice, you'd need to properly tokenize
                # and compute log probabilities for each turn
                prime_log_probs = self._compute_log_probs(self.prime_model, responses)
                ref_log_probs = self._compute_log_probs(self.ref_model, responses)
                
                # Compute KL divergence as process reward
                kl_div = prime_log_probs - ref_log_probs
                process_reward = kl_div * self.beta
                process_rewards.append(process_reward)
        
        return torch.stack(process_rewards) if process_rewards else torch.zeros(1, device=data.batch["responses"].device)
    
    def _extract_turn_responses(self, data: DataProto) -> List[List[str]]:
        """
        Extract turn-level responses from the dialogue history.
        
        Args:
            data: DataProto containing dialogue data
            
        Returns:
            List[List[str]]: List of turn responses for each dialogue
        """
        # This is a simplified extraction - you'll need to adapt based on your data format
        turn_responses = []
        
        # Extract from dialogue history if available
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
                response_text = self.tokenizer.decode(responses[i], skip_special_tokens=True)
                # Simple split by turn markers - adapt based on your format
                turns = response_text.split("<|im_end|>") if "<|im_end|>" in response_text else [response_text]
                turn_responses.append([turn.strip() for turn in turns if turn.strip()])
        
        return turn_responses
    
    def _compute_log_probs(self, model, responses: List[str]) -> torch.Tensor:
        """
        Compute log probabilities for responses using the given model.
        
        Args:
            model: The model to compute log probabilities with
            responses: List of response strings
            
        Returns:
            torch.Tensor: Log probabilities for each response
        """
        # This is a placeholder - implement based on your model interface
        # You'll need to tokenize responses and compute log probabilities
        log_probs = []
        for response in responses:
            # Tokenize and compute log probability
            tokens = self.tokenizer.encode(response, return_tensors="pt")
            with torch.no_grad():
                outputs = model(tokens)
                log_probs.append(outputs.logits.log_softmax(dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1).sum())
        
        return torch.stack(log_probs) if log_probs else torch.zeros(1)
    
    def compute_combined_grpo_advantage(self, data: DataProto, outcome_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute combined GRPO advantage using both outcome and process rewards.
        
        Args:
            data: DataProto containing dialogue data
            outcome_rewards: torch.Tensor of outcome rewards (final scores)
            
        Returns:
            torch.Tensor: Combined advantages for each turn
        """
        # Compute process rewards
        process_rewards = self.compute_turn_level_process_rewards(data)
        
        # Get group information for GRPO
        group_ids = data.non_tensor_batch.get("group_ids", np.arange(len(outcome_rewards)))
        
        # Compute GRPO advantages for outcome rewards
        outcome_advantages = self._compute_grpo_advantage(outcome_rewards, group_ids, "outcome")
        
        # Compute GRPO advantages for process rewards
        process_advantages = self._compute_grpo_advantage(process_rewards, group_ids, "process")
        
        # Combine advantages according to the formula
        combined_advantages = outcome_advantages + self.lambda_1 * process_advantages
        
        return combined_advantages
    
    def _compute_grpo_advantage(self, rewards: torch.Tensor, group_ids: np.ndarray, reward_type: str) -> torch.Tensor:
        """
        Compute GRPO advantage for given rewards.
        
        Args:
            rewards: torch.Tensor of rewards
            group_ids: Group IDs for GRPO grouping
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
            if self.config.algorithm.get("norm_adv_by_std_in_grpo", True):
                advantages[i] = (rewards[i] - id2mean[group_id]) / (id2std[group_id] + 1e-6)
            else:
                advantages[i] = rewards[i] - id2mean[group_id]
        
        return advantages
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Main interface for computing combined rewards.
        
        Args:
            data: DataProto containing dialogue data
            return_dict: Whether to return as dictionary
            
        Returns:
            Combined reward tensor or dictionary
        """
        # Extract outcome rewards (final scores from judge LLM)
        outcome_rewards = self._extract_outcome_rewards(data)
        
        # Compute combined advantages
        combined_advantages = self.compute_combined_grpo_advantage(data, outcome_rewards)
        
        # Convert to token-level rewards
        token_level_rewards = self._convert_to_token_level(combined_advantages, data)
        
        if return_dict:
            return {
                "reward_tensor": token_level_rewards,
                "reward_extra_info": {
                    "outcome_rewards": outcome_rewards.cpu().numpy().tolist(),
                    "combined_advantages": combined_advantages.cpu().numpy().tolist(),
                }
            }
        else:
            return token_level_rewards
    
    def _extract_outcome_rewards(self, data: DataProto) -> torch.Tensor:
        """
        Extract outcome rewards from the data.
        
        Args:
            data: DataProto containing dialogue data
            
        Returns:
            torch.Tensor: Outcome rewards for each dialogue
        """
        # Extract final scores from the dialogue history
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
    
    def _convert_to_token_level(self, turn_advantages: torch.Tensor, data: DataProto) -> torch.Tensor:
        """
        Convert turn-level advantages to token-level rewards.
        
        Args:
            turn_advantages: Turn-level advantages
            data: DataProto containing dialogue data
            
        Returns:
            torch.Tensor: Token-level rewards
        """
        # Get response mask to identify valid tokens
        response_mask = data.batch.get("response_mask", data.batch.get("loss_mask"))
        
        # Create token-level rewards tensor
        token_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
        
        # Distribute turn advantages to tokens in each turn
        for i, advantage in enumerate(turn_advantages):
            if i < token_rewards.shape[0]:
                # Find turn boundaries (EOS tokens or turn markers)
                turn_boundaries = self._find_turn_boundaries(data, i)
                
                # Distribute advantage across tokens in each turn
                for turn_start, turn_end in turn_boundaries:
                    if turn_start < turn_end and turn_start < token_rewards.shape[1]:
                        # Distribute advantage evenly across the turn
                        turn_length = min(turn_end - turn_start, token_rewards.shape[1] - turn_start)
                        if turn_length > 0:
                            token_rewards[i, turn_start:turn_start + turn_length] = advantage / turn_length
        
        return token_rewards
    
    def _find_turn_boundaries(self, data: DataProto, sample_idx: int) -> List[Tuple[int, int]]:
        """
        Find turn boundaries in the dialogue.
        
        Args:
            data: DataProto containing dialogue data
            sample_idx: Index of the sample
            
        Returns:
            List[Tuple[int, int]]: List of (start, end) positions for each turn
        """
        # This is a simplified implementation - adapt based on your tokenization
        response_tokens = data.batch["responses"][sample_idx]
        attention_mask = data.batch["attention_mask"][sample_idx]
        
        # Find EOS tokens or turn markers
        turn_boundaries = []
        current_start = 0
        
        for i, token in enumerate(response_tokens):
            if token == self.tokenizer.eos_token_id or token == self.tokenizer.encode("<|im_end|>")[0]:
                turn_boundaries.append((current_start, i + 1))
                current_start = i + 1
        
        # Add final turn if there are remaining tokens
        if current_start < len(response_tokens):
            turn_boundaries.append((current_start, len(response_tokens)))
        
        return turn_boundaries