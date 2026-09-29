#!/usr/bin/env bash

# By default, evaluate standard first and JBB second:
#   RAGEN_LLM_ROOT=/path/to/models bash eval_run.sh
#
# Other forms:
#   bash eval_run.sh all llama [attacker_dir]
#   bash eval_run.sh standard gemma [attacker_dir]
#   bash eval_run.sh jbb qwen [attacker_dir]

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: RAGEN_LLM_ROOT=/path/to/models bash eval_run.sh [mode] [victim] [attacker_dir]

mode (default: all):
  all       run standard and then JBB evaluation
  standard  use config/_7_jailbreak_eval.yaml
  jbb       use config/_7_jailbreak_eval_jbb.yaml

victim (default: qwen):
  gemma     gemma-2-9b-it
  llama     Llama-3.1-8B-Instruct
  mistral   Mistral-7B-Instruct-v0.3
  qwen      Qwen2.5-7B-Instruct

Optional environment variables:
  ENV_GPU=0       GPU for the victim vLLM server
  JUDGER_GPU=1    GPU for the HarmBench judger vLLM server
  ENV_PORT=9003   victim vLLM port
  JUDGER_PORT=9004
  EVAL_TIMES=3
  TRAINING_STEPS  override the training-step value passed to eval_mine.py
EOF
}

is_victim() {
  case "${1,,}" in
    gemma|gemma-2-9b|gemma-2-9b-it|llama|llama-31-8b|llama-3.1-8b|llama-3.1-8b-instruct|mistral|mistral-7b-v03|mistral-7b-instruct-v0.3|qwen|qwen-25-7b|qwen2.5-7b|qwen2.5-7b-instruct) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

# Keep the previous mode-first form, while also accepting `eval_run.sh qwen`.
mode="all"
victim_arg="qwen"
attacker_arg=""
if [[ $# -ge 1 ]]; then
  case "${1,,}" in
    all|standard|normal|eval|jbb)
      mode="${1,,}"
      if [[ $# -ge 2 ]] && is_victim "$2"; then
        victim_arg="$2"
        attacker_arg="${3:-}"
      else
        # Backward-compatible form: eval_run.sh standard [attacker_dir]
        attacker_arg="${2:-}"
      fi
      ;;
    *)
      if is_victim "$1"; then
        victim_arg="$1"
        attacker_arg="${2:-}"
      else
        echo "Unknown mode or victim: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
fi

case "$mode" in
  all)
    eval_types=(standard jbb)
    ;;
  standard|normal|eval)
    eval_types=(standard)
    ;;
  jbb)
    eval_types=(jbb)
    ;;
  *)
    echo "Unknown evaluation mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac

case "${victim_arg,,}" in
  gemma|gemma-2-9b|gemma-2-9b-it)
    victim_key="gemma"
    victim_ckpt_label="gemma-2-9b"
    target_model_label="gemma-2-9b-it"
    env_model_dir="gemma-2-9b-it"
    env_max_len=8192
    default_training_steps=260
    default_attacker_dir="./checkpoints/jailbreak_grpo/new_GRPO_Heuristic_attack_Qwen3B_victim_gemma-2-9b_classifier_Llama2_hlambda_01_threshold_09_steps_260_lr_1e-6_kl_coef_001_entropy_coef_001"
    ;;
  llama|llama-31-8b|llama-3.1-8b|llama-3.1-8b-instruct)
    victim_key="llama"
    victim_ckpt_label="Llama-31-8B"
    target_model_label="Llama-3.1-8B-Instruct"
    env_model_dir="Llama-3.1-8B-Instruct"
    env_max_len=13312
    default_training_steps=260
    default_attacker_dir="./checkpoints/jailbreak_grpo/new_GRPO_Heuristic_attack_Qwen3B_victim_Llama-31-8B_classifier_Llama2_hlambda_01_threshold_09_steps_260_lr_1e-6_kl_coef_001_entropy_coef_001"
    ;;
  mistral|mistral-7b-v03|mistral-7b-instruct-v0.3)
    victim_key="mistral"
    victim_ckpt_label="Mistral-7B-v03"
    target_model_label="Mistral-7B-Instruct-v0.3"
    env_model_dir="Mistral-7B-Instruct-v0.3"
    env_max_len=13312
    default_training_steps=260
    default_attacker_dir="./checkpoints/jailbreak_grpo/new_GRPO_Heuristic_attack_Qwen3B_victim_Mistral-7B-v03_classifier_Llama2_hlambda_01_threshold_09_steps_260_lr_1e-6_kl_coef_001_entropy_coef_001"
    ;;
  qwen|qwen-25-7b|qwen2.5-7b|qwen2.5-7b-instruct)
    victim_key="qwen"
    victim_ckpt_label="Qwen-25-7B"
    target_model_label="Qwen2.5-7B-Instruct"
    env_model_dir="Qwen2.5-7B-Instruct"
    env_max_len=13312
    default_training_steps=390
    default_attacker_dir="./checkpoints/jailbreak_grpo/new_GRPO_Heuristic_attack_Qwen3B_victim_Qwen-25-7B_classifier_Llama2_hlambda_01_threshold_09_steps_260_lr_1e-6_kl_coef_001_entropy_coef_001"
    ;;
esac

: "${RAGEN_LLM_ROOT:?Set RAGEN_LLM_ROOT before running this script}"
command -v nc >/dev/null 2>&1 || {
  echo "The 'nc' command is required to check the vLLM ports." >&2
  exit 1
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

attacker_dir="${attacker_arg:-${ATTACKER_DIR:-$default_attacker_dir}}"
env_llm_path="$RAGEN_LLM_ROOT/$env_model_dir"
judger_llm_path="$RAGEN_LLM_ROOT/HarmBench-Llama-2-13b-cls"
model_path="$RAGEN_LLM_ROOT/Qwen2.5-3B-Instruct"

for path in "$attacker_dir" "$env_llm_path" "$judger_llm_path" "$model_path"; do
  [[ -d "$path" ]] || {
    echo "Directory does not exist: $path" >&2
    exit 1
  }
done

env_gpu="${ENV_GPU:-0}"
judger_gpu="${JUDGER_GPU:-1}"
env_llm_port="${ENV_PORT:-9003}"
judger_llm_port="${JUDGER_PORT:-9004}"
eval_times="${EVAL_TIMES:-3}"
training_steps="${TRAINING_STEPS:-$default_training_steps}"
env_llm_base_url="http://localhost:$env_llm_port/v1"
judger_llm_base_url="http://localhost:$judger_llm_port/v1"

for port in "$env_llm_port" "$judger_llm_port"; do
  if nc -z localhost "$port" 2>/dev/null; then
    echo "Port $port is already in use; choose another port." >&2
    exit 1
  fi
done

ulimit -u 65535 2>/dev/null || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_NUM_CPUS=12 RAY_DISABLE_MEMORY_MONITOR=1 RAY_DEDUP_LOGS=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p nohup_logs/eval_logs
env_llm_pid=""
judger_llm_pid=""

cleanup() {
  [[ -n "$env_llm_pid" ]] && kill "$env_llm_pid" 2>/dev/null || true
  [[ -n "$judger_llm_pid" ]] && kill "$judger_llm_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CUDA_VISIBLE_DEVICES="$judger_gpu" python -m vllm.entrypoints.openai.api_server \
  --model "$judger_llm_path" --port "$judger_llm_port" --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 --max-model-len 2048 \
  --max-num-seqs 256 --max-num-batched-tokens 51200 \
  --tensor-parallel-size 1 \
  >"nohup_logs/judger_llm_eval.log" 2>&1 &
judger_llm_pid=$!

CUDA_VISIBLE_DEVICES="$env_gpu" python -m vllm.entrypoints.openai.api_server \
  --model "$env_llm_path" --port "$env_llm_port" --host 0.0.0.0 \
  --gpu-memory-utilization 0.95 --max-model-len "$env_max_len" \
  --max-num-seqs 256 --max-num-batched-tokens 51200 \
  --tensor-parallel-size 1 \
  >"nohup_logs/env_llm_eval.log" 2>&1 &
env_llm_pid=$!

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

run_eval() {
  local eval_type="$1" config_name experiment_prefix experiment_name eval_log
  case "$eval_type" in
    standard)
      config_name="_7_jailbreak_eval.yaml"
      experiment_prefix="eval"
      export TROJAIL_VAL_DATA_PATHS="./data/harmbench_train-00000-of-00001.parquet:./data/strongreject_dataset.csv"
      ;;
    jbb)
      config_name="_7_jailbreak_eval_jbb.yaml"
      experiment_prefix="eval_jbb"
      if [[ ! -f "./data/jbb_harmful-behaviors.csv" ]]; then
        echo "JBB dataset not found: ./data/jbb_harmful-behaviors.csv" >&2
        return 1
      fi
      export TROJAIL_VAL_DATA_PATHS="./data/jbb_harmful-behaviors.csv"
      ;;
  esac

  experiment_name="${experiment_prefix}_new_GRPO_Heuristic_attack_Qwen3B_victim_${victim_ckpt_label}_Target_${target_model_label}_20260704_attempt2"
  eval_log="nohup_logs/eval_logs/${experiment_name}.log"
  mkdir -p "tensorboard_log/$experiment_name"
  echo "Starting $eval_type evaluation for $victim_key..."

  python -u eval_mine.py --config-name "$config_name" \
    "trainer.default_local_dir=$attacker_dir" \
    "trainer.eval_times=$eval_times" \
    "model_path=$model_path" \
    "env_llm.model_path=$env_llm_path" \
    "judger_llm.model_path=$judger_llm_path" \
    "env_llm.base_url=$env_llm_base_url" \
    "judger_llm.base_url=$judger_llm_base_url" \
    algorithm.heuristic_process_adv_lambda=0.1 \
    "experiment_name=$experiment_name" \
    "trainer.total_training_steps=$training_steps" trainer.test_freq=10 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    >"$eval_log" 2>&1

  echo "Finished $eval_type evaluation: $experiment_name"
}

for eval_type in "${eval_types[@]}"; do
  run_eval "$eval_type"
done

echo "All requested evaluations finished."
