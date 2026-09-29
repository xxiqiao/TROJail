export VLLM_WORKER_MULTIPROC_METHOD=spawn

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --model ../LLMs/Llama-Guard-3-8B \
  --port 8002 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --tensor-parallel-size 1
  # --max-model-len 32768 \
  # --served-model-name "judger_llm"

    # --model-name judger_llm \