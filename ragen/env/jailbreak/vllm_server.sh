export VLLM_WORKER_MULTIPROC_METHOD=spawn
: "${RAGEN_LLM_ROOT:?Set RAGEN_LLM_ROOT before running this script}"
env_model="${ENV_LLM_MODEL:-$RAGEN_LLM_ROOT/Llama-3.2-3B-Instruct}"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$env_model" \
  --port 8001 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 13312 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 102400 \
  --tensor-parallel-size 1 \
  --enable-prefix-caching

# CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#   --model ../LLMs/Llama-3.1-8B-Instruct \
#   --port 8001 \
#   --host 0.0.0.0 \
#   --gpu-memory-utilization 0.9 \
#   --max-model-len 13312 \
#   --max-num-seqs 256 \
#   --max-num-batched-tokens 102400 \
#   --tensor-parallel-size 2 \
#   --enable-prefix-caching
#   # --max-model-len 8192 \
#   # --max-model-len 32768 \
#   # --served-model-name "env_llm"

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
#   --model ../LLMs/Llama-3.1-8B-Instruct \
#   --port 8001 \
#   --host 0.0.0.0 \
#   --gpu-memory-utilization 0.9 \
#   --max-model-len 13312 \
#   --max-num-seqs 256 \
#   --max-num-batched-tokens 204800 \
#   --tensor-parallel-size 4

# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#   --model ../LLMs/Llama-3.1-8B-Instruct \
#   --port 8001 \
#   --host 0.0.0.0 \
#   --gpu-memory-utilization 0.9 \
#   --max-model-len 13312 \
#   --max-num-seqs 256 \
#   --max-num-batched-tokens 10240 \
#   --tensor-parallel-size 1
