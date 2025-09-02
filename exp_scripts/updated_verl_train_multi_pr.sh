set -x
export HF_ENDPOINT=https://hf-mirror.com

eval "$(conda shell.bash hook)"
conda activate rl

# NOTE: change to your root dir
ROOT=/data/RLAMG

# ray stop 

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export http_proxy="http://100.78.52.155:7897"
export https_proxy="http://100.78.52.155:7897"

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

export MODEL_PATH=/data1/models/Qwen2.5-1.5B-Instruct
export DATA_DIR=$ROOT/data/

export WANDB_PROJECT="Adaptive_Multi_LUFFY_PR_newverl"

# export EXP_NAME=Multi_LUFFY_TEST_4_clip_always
# export EXP_NAME=Adaptive_Multi_LUFFY_2_LongCoT
# export EXP_NAME="always_multi_luffy_2_longcot"
# export EXP_NAME="adaptive_multi_luffy_2_longcot_threshold_2"
export EXP_NAME="adaptive_multi_2_Qwen2.5-1.5B_longcot"

cd $ROOT/rlamg/verl/

data_size=8491

output_dir="/data1/RL_output/Qwen2.5-1.5B/checkpoints/$WANDB_PROJECT/$EXP_NAME"

# Train over a single node, 8 A100-80GB GPUs.
CUDA_VISIBLE_DEVICES=1 python3 -m verl.adaptive_mix_src_pr.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/openr1_multi_longcot_pr.parquet \
    data.val_files=$DATA_DIR/valid_math_all.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.num_off_policy_targets=2 \
    data.max_available_targets=4 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9316 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.rollout.injection_strategy='adaptive' \
    actor_rollout_ref.rollout.target_selection_strategy='best_k' \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_prefix_len=8192 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_multitask_loss=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1" \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    reward_model.reward_manager_shaping_function_name=threshold_0 \
    reward_model.compute_score_name=mean_exp_log_softmax \
    reward_model.repetition_penalty=True \
    reward_model.off_policy_reward_manager=prob \
    reward_model.val_reward_manager=naive \
    reward_model.format_mode=R1 \
    reward_model.format_coefficient=0.0 \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=False \
    data.reward_impl_version=5 \
    data.val_reward_impl_version=5 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$output_dir \
    trainer.total_epochs=5 "${@:1}"
