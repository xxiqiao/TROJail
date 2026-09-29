import math
from typing import List, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class VllmLocalClient:
    def __init__(self, config, mode="env_llm"):



        self.config = config
        if mode == "env_llm":
            model_path = config.env_llm.model_path
            vllm_cfg = config.env_llm
        elif mode == "judger_llm":
            model_path = config.judger_llm.model_path
            vllm_cfg = config.judger_llm
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model_path,
            dtype=getattr(vllm_cfg, "dtype", "bfloat16"),
            gpu_memory_utilization=getattr(vllm_cfg, "gpu_memory_utilization", 0.9),
            max_model_len=getattr(vllm_cfg, "max_model_len", 32768),
            tensor_parallel_size=getattr(vllm_cfg, "tensor_parallel_size", 1),
            trust_remote_code=True,
        )
        self.default_temperature = getattr(vllm_cfg, "temperature", 0.7)
        self.default_max_tokens = getattr(vllm_cfg, "max_tokens", 4096)

    def _messages_to_prompts(self, batch_messages: List[List[dict]]) -> List[str]:

        prompts = []
        for messages in batch_messages:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            prompts.append(prompt)
        return prompts

    def chat_batch(self, batch_messages: List[List[dict]], max_tokens: Optional[int]=None, temperature: Optional[float]=None, stop=None) -> List[str]:
        prompts = self._messages_to_prompts(batch_messages)
        sampling_params = SamplingParams(
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature if temperature is not None else self.default_temperature,
            stop=stop,
        )
        outputs = self.llm.generate(prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]

    def get_logprobs_batch(self, batch_messages: List[List[dict]], max_tokens: int=1, logprobs: int=2):
        prompts = self._messages_to_prompts(batch_messages)
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            logprobs=logprobs,
        )
        outputs = self.llm.generate(prompts, sampling_params)


        results = []
        for output in outputs:
            logprobs_dict = output.outputs[0].logprobs[0]  # {token_id: logprob, ...}
            results.append(logprobs_dict)
        return results

    def get_unsafe_probs(self, prompts, responses, unsafe_token="unsafe"):

        system_prompts = []
        for x, y in zip(prompts, responses):
            chat = [
                {"role": "user", "content": x},
                {"role": "assistant", "content": y}
            ]
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
            system_prompts.append(prompt)
        unsafe_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(unsafe_token))[0]
        sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=2
        )
        llm_outputs = self.llm.generate(system_prompts, sampling_params)
        log_rewards = []
        for output in llm_outputs:
            log_prob = output.outputs[0].logprobs[0][unsafe_id]
            if not isinstance(log_prob, float):
                log_prob = log_prob.logprob
            log_rewards.append(log_prob)
        return [math.exp(logp) for logp in log_rewards]