python ../RAGEN/verl/scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path ../LLMs/Qwen2.5-0.5B-Instruct \
    --local_dir ../RAGEN/checkpoints/jailbreak_grpo/GRPO_PRIME_token_attack_Qwen05B_victim_Llama-32-3B_classifier_Llama2_plambda_001_threshold_09_steps_620_lr_1e-6_1e-4_kl_coef_001_entropy_coef_001/best/actor \
    --target_dir ../RAGEN/checkpoints/hf_model
