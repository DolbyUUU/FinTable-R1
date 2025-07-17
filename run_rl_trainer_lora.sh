#!/bin/bash

set -x

# Hugging Face access token
export HF_HUB_TOKEN="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"
export HUGGINGFACE_HUB_TOKEN="$HF_HUB_TOKEN"
# export HF_HOME="/home/data/base_models"

# Use virtual environment
source /home/yu/venv-verl/bin/activate

# Environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS
export MKL_THREADING_LAYER=GNU
export TORCH_CUDA_ALLOW_TF32=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# wandb configuration
export WANDB_MODE=online
# export WANDB_API_KEY=17dc79f602cb38784f4e5f10986e3a3355f54315
# export WANDB_API_KEY=c2fe654fd0527d4fad92e03cdc1e3f59b9a20595
export WANDB_BASE_URL=https://api.wandb.ai

export PYTHONPATH=/home/yu/FinTable-R1/verl:$PYTHONPATH

# Save the current working directory
CURRENT_DIR=$(pwd)

# Change directory to the data preprocessing folder
cd ./data_preprocess_finben || { echo "Failed to change directory to ./data_preprocess_finben"; exit 1; }

# Run data preprocessing
echo "Running data preprocessing..."
python3 finben_data_preprocessing.py --local_dir /home/yu/FinTable-R1/data_preprocess_finben

# Return to the original directory
cd "$CURRENT_DIR" || { echo "Failed to return to $CURRENT_DIR"; exit 1; }

# Default parameters
DATA_DIR="data_preprocess_finben/cra-lendingclub"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROJECT_NAME="FinTable-R1"
N_GPUS=4
ROLLOUT_TP_SIZE=4

# Read reward parameters from JSON file
REWARD_CONFIG_PATH="config.json"
REWARD_PARAMS=$(jq -r 'to_entries | map(.value) | join("-")' $REWARD_CONFIG_PATH)

# Build experiment name
DATASET_NAME=$(basename "$DATA_DIR")
EXPERIMENT_NAME="${DATASET_NAME}-lora-${REWARD_PARAMS}-$(date +%Y%m%d%H%M)"
echo "Experiment Name: $EXPERIMENT_NAME"

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR="model_checkpoints/${EXPERIMENT_NAME}"

# Clear GPU memory cache
echo "Clearing GPU memory cache..."
python3 -c "import torch; torch.cuda.empty_cache()"
echo "GPU memory cache cleared."

# Start training from scratch
echo "Starting the training process..."
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    +data.data_source=finben \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/validation.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    +actor_rollout_ref.model.use_lora=True \
    +actor_rollout_ref.model.lora_r=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    +actor_rollout_ref.model.lora_dropout=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    +actor_rollout_ref.rollout.val_temperature=0.3 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_local_dir=$DEFAULT_CHECKPOINT_DIR \
    trainer.resume_mode="disable" \
    trainer.total_epochs=20 \
    trainer.val_before_train=True 2>&1 | tee log.log
