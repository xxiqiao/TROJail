#!/usr/bin/env bash
#An example for GPU job.
#SBATCH -J Q-L-oh-ne
#SBATCH -o ./log_jobs/job-%j-ne.log
#SBATCH -e ./log_jobs/job-%j-ne.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu_8a100
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
if [[ -n "${RAGEN_CONDA_BASE:-}" ]]; then
  . "$RAGEN_CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${RAGEN_CONDA_ENV:-TROJail}"
fi





export RAY_NUM_CPUS=$SLURM_CPUS_PER_TASK


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


ulimit -n 65535
ulimit -u 65535


export RAY_DISABLE_MEMORY_MONITOR=1



echo This job runs in conda environment $CONDA_DEFAULT_ENV
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export NCCL_IB_DISABLE=1
: "${RAGEN_LLM_ROOT:?Set RAGEN_LLM_ROOT to your local model directory}"
PYTHONPATH="$repo_root/verl:${PYTHONPATH:-}" bash victim_run.sh llama
