export VLLM_WORKER_MULTIPROC_METHOD=spawn
: "${RAGEN_LLM_ROOT:?Set RAGEN_LLM_ROOT before running this script}"
judger_model="${JUDGER_LLM_MODEL:-$RAGEN_LLM_ROOT/HarmBench-Llama-2-13b-cls}"

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model "$judger_model" \
  --port 8002 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 51200 \
  --tensor-parallel-size 1 \
  --enable-prefix-caching
  # --model ../LLMs/HarmBench-Llama-2-13b-cls \
  # --model ../LLMs/HarmBench-Mistral-7b-val-cls \
  # --max-model-len 4096 \
  # --served-model-name "judger_llm"


# CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
#   --model ../LLMs/HarmBench-Llama-2-13b-cls \
#   --port 8002 \
#   --host 0.0.0.0 \
#   --gpu-memory-utilization 0.9 \
#   --max-model-len 2048 \
#   --max-num-seqs 256 \
#   --max-num-batched-tokens 102400 \
#   --tensor-parallel-size 2

# ../LLMs/HarmBench-Mistral-7b-val-cls
# ../LLMs/Llama-3.1-8B-Instruct

# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#   --model ../LLMs/HarmBench-Llama-2-13b-cls \
#   --port 8002 \
#   --host 0.0.0.0 \
#   --gpu-memory-utilization 0.9 \
#   --max-model-len 2048 \
#   --max-num-seqs 256 \
#   --max-num-batched-tokens 51200 \
#   --tensor-parallel-size 1
