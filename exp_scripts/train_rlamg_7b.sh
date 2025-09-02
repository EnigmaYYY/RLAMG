#!/usr/bin/env bash
set -xeuo pipefail

# export no_proxy="127.0.0.1,localhost"
# export NO_PROXY="127.0.0.1,localhost"
# export http_proxy="http://100.78.52.155:7897"
# export https_proxy="http://100.78.52.155:7897"

# 添加 wandb 环境变量
export WANDB_MODE=online
export WANDB_API_KEY="b1c482b1f572b6802f9d14fb96ad380af479ae82"

# export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

# project_name='MIX-DAPO'
project_name='RLAMG'
exp_name='Qwen2.5-7b-Ins-rlamg-2shortcot'

# Paths
HOME=/home/yxy/RLAMG/rlamg
RAY_DATA_HOME="${HOME}/verl"
cd $HOME/verl
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
MODEL_PATH="/mnt/data/yxy/models/Qwen2.5-7B-Instruct"
CKPTS_DIR="/mnt/data/yxy/RL_output/Qwen2.5-7B-Ins/checkpoints/${project_name}/${exp_name}"
TRAIN_FILE="/home/yxy/RLAMG/data/openr1_multi_shortcot.parquet"
TEST_FILE="/home/yxy/RLAMG/data/valid.parquet"

# MIX 
num_off_policy_targets=2
max_available_targets=4
use_sft_multitask_loss=False
use_off_policy_loss=True
off_policy_normalize=False
off_policy_reshape="p_div_p_0.1"
off_policy_loss_impl='seq'
loss_remove_token_mean=True
loss_remove_clip=False
injection_strategy='adaptive'
target_selection_strategy='best_k'
prefix_strategy='random'

# reward
reward_impl_version=5
val_reward_impl_version=5
off_policy_reward_manager='prob'
reward_manager='naive'
val_reward_manager='naive'

adv_estimator=grpo

clip_ratio_low=0.2
# clip_ratio_high=0.28
clip_ratio_high=0.2

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=128
val_prompt_bsz=512
n_resp_per_prompt=8
train_prompt_mini_bsz=64

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=1
NGPUS_PER_NODE=4

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.6
val_top_p=1.0

filter_groups_enabled=False
filter_groups_metric='seq_reward'
max_num_gen_batches=10

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
#actor_ppo_max_token_len=$(((max_response_length) * 4 ))
#infer_ppo_max_token_len=$(((max_response_length) * 4 ))

offload=True
gen_tp=2
fsdp_size=-1

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# reference run wandb: https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/runs/ow47vvon?nw=nwusertongyuxuan361

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.adaptive_mix_src.main_mix_dapo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz} \
    data.shuffle=True \
    data.num_off_policy_targets=${num_off_policy_targets} \
    data.max_available_targets=${max_available_targets} \
    data.reward_impl_version=${reward_impl_version} \
    data.val_reward_impl_version=${val_reward_impl_version} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.filter_groups.enable=${filter_groups_enabled} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.max_prefix_len=${max_response_length} \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.use_sft_multitask_loss=${use_sft_multitask_loss} \
    actor_rollout_ref.actor.use_off_policy_loss=${use_off_policy_loss} \
    actor_rollout_ref.actor.off_policy_normalize=${off_policy_normalize} \
    actor_rollout_ref.actor.off_policy_reshape=${off_policy_reshape} \
    actor_rollout_ref.actor.off_policy_loss_impl=${off_policy_loss_impl} \
    actor_rollout_ref.actor.loss_remove_token_mean=${loss_remove_token_mean} \
    actor_rollout_ref.actor.loss_remove_clip=${loss_remove_clip} \
    actor_rollout_ref.rollout.injection_strategy=${injection_strategy} \
    actor_rollout_ref.rollout.target_selection_strategy=${target_selection_strategy} \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    reward_model.reward_manager_shaping_function_name=threshold_0 \
    reward_model.compute_score_name=mean_exp_log_softmax \
    reward_model.repetition_penalty=True \
    reward_model.off_policy_reward_manager=${off_policy_reward_manager} \
    reward_model.val_reward_manager=${val_reward_manager} \
    reward_model.format_mode=R1_nothink \
    reward_model.format_coefficient=0.1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=30 \
    trainer.total_training_steps=510 \
    +trainer.max_optim_to_keep=2 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
