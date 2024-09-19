#!/bin/sh

rank=4
alpha=8
gpuid=0

model_p_or_n=yahma/llama-7b-hf

model_path=trained_models/mlora-r$rank-a$alpha-3e4
results_path=results/mlora-r$rank-a$alpha-3e4

mkdir -p $model_path
mkdir -p $results_path

# MLoRA: --use_lora_router
# MixLoRA: --use_lora_router --use_lora_router_mixer


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