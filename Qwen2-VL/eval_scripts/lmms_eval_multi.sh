#!/bin/bash

# 参数配置区域
# tasks=("gqa" "mmbench_en" "mmbench_cn" "mme" "pope" "scienceqa_img" "textvqa")  # 示例任务列表
# pruned_layers=(2 3 5)     # 剪枝层数候选
# reduction_ratios=(0.2 0.3 0.5) # 压缩率候选

tasks=("scienceqa_full")
pruned_layers=(2 5 10)
reduction_ratios=(0.2 0.3)

for task in "${tasks[@]}"; do
  for pruned_layer in "${pruned_layers[@]}"; do
    for reduction_ratio in "${reduction_ratios[@]}"; do
      
      # 打印当前参数组合
      echo "========================================"
      echo "当前参数组合:"
      echo "Task: $task"
      echo "Pruned Layer: $pruned_layer"
      echo "Reduction Ratio: $reduction_ratio"
      echo "========================================"

      model_id="Qwen/Qwen2-VL-7B-Instruct"
      model_name="Qwen2-VL-7B-Instruct"
      output_path="./logs/${model_name}/${task}/pruned_${pruned_layer}_ratio_${reduction_ratio}/"
      mkdir -p "$output_path"

      Sparse=True
      image_token_start_index=0
      image_token_length=0
      max_num_trunction=128
      pivot_image_token=4
      pivot_text_token=4
      random_choose=False

      python3 -m accelerate.commands.launch \
          --num_processes=2 \
          --main_process_port 50008 \
          -m lmms_eval \
          --model qwen2_vl_dart_vit \
          --model_args pretrained=$model_id,device_map=cuda,use_flash_attention_2=True,Sparse=$Sparse,pruned_layer=$pruned_layer,image_token_start_index=$image_token_start_index,image_token_length=$image_token_length,max_num_trunction=$max_num_trunction,reduction_ratio=$reduction_ratio,pivot_image_token=$pivot_image_token,pivot_text_token=$pivot_text_token,random_choose=$random_choose\
          --tasks "${task}" \
          --batch_size 1 \
          --log_samples \
          --output_path "$output_path"

      # 添加间隔时间避免端口冲突
      sleep 10
    done
  done
done