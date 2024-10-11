#!/bin/sh

rank=4
alpha=8
gpuid=0
micro_batch_size=4

model_p_or_n=yahma/llama-7b-hf

model_path=trained_models/mlora-r$rank-a$alpha-5e4-h4-epoch3
results_path=results/mlora-r$rank-a$alpha-5e4-h4-epoch3

mkdir -p $model_path
mkdir -p $results_path

# MLoRA: --use_lora_router
# MixLoRA: --use_lora_router --use_lora_router_mixer

CUDA_VISIBLE_DEVICES=$gpuid python -u finetune.py \
  --base_model $model_p_or_n \
  --data_path 'ft_training_set/commonsense_170k.json' \
  --output_dir $model_path \
  --batch_size 16 \
  --micro_batch_size $micro_batch_size \
  --num_epochs 3 \
  --learning_rate 5e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  --lora_r $rank \
  --lora_alpha $alpha \
  --use_lora_router \
  --target_modules "["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]"


for ds in ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq hellaswag
do
  CUDA_VISIBLE_DEVICES=$gpuid python -u commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset $ds \
    --batch_size 1 \
    --base_model $model_p_or_n \
    --lora_weights $model_path \
    --save_dir $results_path
done