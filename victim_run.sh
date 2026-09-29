#!/usr/bin/env bash

# Usage: RAGEN_LLM_ROOT=/path/to/models bash victim_run.sh <victim>
# Example: bash victim_run.sh qwen

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: RAGEN_LLM_ROOT=/path/to/models bash victim_run.sh <victim>

victim:
  gemma       gemma-2-9b-it
  llama       Llama-3.1-8B-Instruct
  mistral     Mistral-7B-Instruct-v0.3
  qwen        Qwen2.5-7B-Instruct

Optional: ENV_GPU=0 JUDGER_GPU=1 bash victim_run.sh <victim>
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

case "${1,,}" in
  gemma|gemma-2-9b|gemma-2-9b-it)
    victim_key="gemma"
    victim_label="gemma-2-9b"
    victim_dir="gemma-2-9b-it"
    env_llm_port=9015
    env_max_len=8192
    max_num_seqs=256
    gpu_memory_utilization=0.90
    ;;
  llama|llama-31-8b|llama-3.1-8b|llama-3.1-8b-instruct)
    victim_key="llama"
    victim_label="Llama-31-8B"
    victim_dir="Llama-3.1-8B-Instruct"
    env_llm_port=9003
    env_max_len=13312
    max_num_seqs=64
    gpu_memory_utilization=0.90
    ;;
  mistral|mistral-7b-v03|mistral-7b-instruct-v0.3)
    victim_key="mistral"
    victim_label="Mistral-7B-v03"
    victim_dir="Mistral-7B-Instruct-v0.3"
    env_llm_port=9021
    env_max_len=13312
    max_num_seqs=256
    gpu_memory_utilization=0.95
    ;;
  qwen|qwen-25-7b|qwen2.5-7b|qwen2.5-7b-instruct)
    victim_key="qwen"
    victim_label="Qwen-25-7B"
    victim_dir="Qwen2.5-7B-Instruct"
    env_llm_port=9009
    env_max_len=13312
    max_num_seqs=256
    gpu_memory_utilization=0.95
    ;;
  *)
    echo "Unknown victim: $1" >&2
    usage >&2
    exit 2
    ;;
esac

: "${RAGEN_LLM_ROOT:?Set RAGEN_LLM_ROOT before running this script}"
command -v nc >/dev/null 2>&1 || {
  echo "The 'nc' command is required to check the vLLM ports." >&2
  exit 1
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

env_llm_path="$RAGEN_LLM_ROOT/$victim_dir"
judger_llm_path="$RAGEN_LLM_ROOT/HarmBench-Llama-2-13b-cls"
model_path="$RAGEN_LLM_ROOT/Qwen2.5-3B-Instruct"

for path in "$env_llm_path" "$judger_llm_path" "$model_path"; do
  [[ -d "$path" ]] || {
    echo "Model directory does not exist: $path" >&2
    exit 1
  }
done

judger_llm_port=$((env_llm_port + 1))
env_llm_base_url="http://localhost:$env_llm_port/v1"
judger_llm_base_url="http://localhost:$judger_llm_port/v1"
env_gpu="${ENV_GPU:-0}"
judger_gpu="${JUDGER_GPU:-1}"

# Apply the process and thread limits to every victim model.
echo "Current ulimit -u (Max Processes/Threads): $(ulimit -u)"
ulimit -u 65535 2>/dev/null || echo "Cannot increase ulimit, sticking to default."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_NUM_CPUS=12 RAY_DISABLE_MEMORY_MONITOR=1
export RAY_DEDUP_LOGS=0 RAY_disable_filesystem_check=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

experiment_name="new_GRPO_Heuristic_attack_Qwen3B_victim_${victim_label}_classifier_Llama2_hlambda_01_threshold_09_steps_260_lr_1e-6_kl_coef_001_entropy_coef_001"
mkdir -p nohup_logs/run_logs "tensorboard_log/$experiment_name"

env_log="nohup_logs/${victim_key}_heuristic_env.log"
judger_log="nohup_logs/${victim_key}_heuristic_judger.log"
train_log="nohup_logs/run_logs/${experiment_name}.log"
env_llm_pid=""
judger_llm_pid=""

cleanup() {
  [[ -n "$env_llm_pid" ]] && kill "$env_llm_pid" 2>/dev/null || true
  [[ -n "$judger_llm_pid" ]] && kill "$judger_llm_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$env_gpu" python -m vllm.entrypoints.openai.api_server \
  --model "$env_llm_path" --port "$env_llm_port" --host 0.0.0.0 \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --max-model-len "$env_max_len" --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens 51200 --tensor-parallel-size 1 \
  >"$env_log" 2>&1 &
env_llm_pid=$!

CUDA_VISIBLE_DEVICES="$judger_gpu" python -m vllm.entrypoints.openai.api_server \
  --model "$judger_llm_path" --port "$judger_llm_port" --host 0.0.0.0 \
  --gpu-memory-utilization "$gpu_memory_utilization" --max-model-len 2048 \
  --max-num-seqs "$max_num_seqs" --max-num-batched-tokens 51200 \
  --tensor-parallel-size 1 \
  >"$judger_log" 2>&1 &
judger_llm_pid=$!

wait_for_port() {
  local port="$1" pid="$2"
  for ((attempt = 1; attempt <= 120; attempt++)); do
    nc -z localhost "$port" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || {
      echo "vLLM server for port $port exited; check its log." >&2
      return 1
    }
    sleep 5
  done
  echo "Timed out waiting for port $port." >&2
  return 1
}

echo "Waiting for vLLM servers ($env_llm_port, $judger_llm_port)..."
wait_for_port "$env_llm_port" "$env_llm_pid"
wait_for_port "$judger_llm_port" "$judger_llm_pid"

python -u train.py --config-name _7_jailbreak_grpo_heuristic.yaml \
  "model_path=$model_path" \
  "env_llm.model_path=$env_llm_path" \
  "judger_llm.model_path=$judger_llm_path" \
  "env_llm.base_url=$env_llm_base_url" \
  "judger_llm.base_url=$judger_llm_base_url" \
  algorithm.heuristic_process_adv_lambda=0.1 \
  "experiment_name=$experiment_name" \
  trainer.total_training_steps=260 trainer.test_freq=10 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  >"$train_log" 2>&1

echo "Training finished: $experiment_name"
