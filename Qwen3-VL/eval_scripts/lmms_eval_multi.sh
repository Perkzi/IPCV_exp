
#!/bin/bash

# ������������
# tasks=("gqa" "mmbench_en" "mmbench_cn" "mme" "pope" "scienceqa_img" "seedbench" "vqav2" "textvqa" "vizwiz_vqa" "ocrbench")  # ʾ�������б�
#        ("mvbench")
# ("gqa" "mmbench_en" "mmbench_cn" "mme" "pope" "seedbench" "textvqa" "vizwiz_vqa" "ocrbench")
# ("mvbench" "videomme" "mlvu" "egoschema")
# pruned_layers=(2 3 5)     # ��֦������ѡ
# reduction_ratios=(0.2 0.3 0.5) # ѹ���ʺ�ѡ

tasks=("gqa"  "mmbench_cn" "mme" "pope" "seedbench" "textvqa" "vizwiz_vqa" "ocrbench")
pruned_layers=(3)
reduction_ratios=(0.93)
vit_pruned_layers=(3)
vit_reduction_ratios=(0.93)

for task in "${tasks[@]}"; do
  for pruned_layer in "${pruned_layers[@]}"; do
    for reduction_ratio in "${reduction_ratios[@]}"; do
      for vit_pruned_layer in "${vit_pruned_layers[@]}"; do
        for vit_reduction_ratio in "${vit_reduction_ratios[@]}"; do
          
          # ��ӡ��ǰ�������
          echo "========================================"
          echo "Current param group:"
          echo "Task: $task"
          echo "Pruned Layer: $pruned_layer"
          echo "Reduction Ratio: $reduction_ratio"
          echo "Vit Pruned Layer: $vit_pruned_layer"
          echo "Vit Reduction Ratio: $vit_reduction_ratio"
          echo "========================================"

          model_id="Qwen/Qwen3-VL-8B-Instruct"
          model_name="Qwen3-VL-8B-Instruct"


          output_path="./logs/${model_name}/${task}/pruned_${pruned_layer}_ratio_${reduction_ratio}/vit_pruned_${vit_pruned_layer}_vit_ratio_${vit_reduction_ratio}/"
          #output_path="./logs/${model_name}/${task}/vit_pruned_${vit_pruned_layer}_vit_ratio_${vit_reduction_ratio}/"
          mkdir -p "$output_path"

          # 对于similarity_kv， Sparse和vit_Sparse都要为True
          # 对于modeling_qwen2_vl_dart_vit_base和其它方法，在主模型prune的Sparse=True,vit_Sparse=False，在vit prune的Sparse=False,vit_Sparse=True
          # 对于vanilla (modeling_qwen2_vl_dart_vit_base)  Sparse和vit_Sparse都要为False
          Sparse=False
          vit_Sparse=False
          image_token_start_index=0
          image_token_length=0
          max_num_trunction=128
          pivot_image_token=4
          pivot_text_token=4

          # 修改在主模型上的剪枝方法
          random_choose=False
          attn_scores_choose=False
          diff_choose=False
          pivot_sim_choose=False

          # 修改在vit上的剪枝方法
          vit_random_choose=False
          vit_attn_scores_choose=False
          vit_diff_choose=False
          vit_pivot_sim_choose=False

          torch_dtype=float16

          method=sparsevlm  # default base ipcv 

          #GPU=0,1,2,3
          #GPU=7

          # 定义log文件名
          log_file="${output_path}/run_detail.log"

          # CUDA_VISIBLE_DEVICES=$GPU python3 -m accelerate.commands.launch \
          #     --num_processes=1 \
          #     --main_process_port 50008 \
          #     -m lmms_eval \
          #     --model qwen3_vl_ipcv \
          #     --model_args pretrained=$model_id,device_map=auto,Sparse=$Sparse,vit_Sparse=$vit_Sparse,pruned_layer=$pruned_layer,vit_pruned_layer=$vit_pruned_layer,image_token_start_index=$image_token_start_index,image_token_length=$image_token_length,max_num_trunction=$max_num_trunction,reduction_ratio=$reduction_ratio,vit_reduction_ratio=$vit_reduction_ratio,pivot_image_token=$pivot_image_token,pivot_text_token=$pivot_text_token,random_choose=$random_choose,attn_scores_choose=$attn_scores_choose,diff_choose=$diff_choose,pivot_sim_choose=$pivot_sim_choose,vit_random_choose=$vit_random_choose,vit_attn_scores_choose=$vit_attn_scores_choose,vit_diff_choose=$vit_diff_choose,vit_pivot_sim_choose=$vit_pivot_sim_choose,torch_dtype=$torch_dtype,method=$method \
          #     --tasks "${task}" \
          #     --batch_size 1 \
          #     --log_samples \
          #     --output_path "$output_path" 2>&1 | tee "$log_file"

          # CUDA_VISIBLE_DEVICES=$GPU python3 -m accelerate.commands.launch \
          #   --num_processes=1 \
          #   --main_process_port 50008 \
          #   -m lmms_eval \
          #   --model qwen3_vl_ipcv \
          #   --model_args pretrained=$model_id,device_map=cuda,Sparse=$Sparse,vit_Sparse=$vit_Sparse,pruned_layer=$pruned_layer,vit_pruned_layer=$vit_pruned_layer,image_token_start_index=$image_token_start_index,image_token_length=$image_token_length,max_num_trunction=$max_num_trunction,reduction_ratio=$reduction_ratio,vit_reduction_ratio=$vit_reduction_ratio,pivot_image_token=$pivot_image_token,pivot_text_token=$pivot_text_token,random_choose=$random_choose,attn_scores_choose=$attn_scores_choose,diff_choose=$diff_choose,pivot_sim_choose=$pivot_sim_choose,vit_random_choose=$vit_random_choose,vit_attn_scores_choose=$vit_attn_scores_choose,vit_diff_choose=$vit_diff_choose,vit_pivot_sim_choose=$vit_pivot_sim_choose,torch_dtype=$torch_dtype,method=$method \
          #   --tasks "${task}" \
          #   --batch_size 1 \
          #   --log_samples \
          #   --output_path "$output_path"

          python3 -m accelerate.commands.launch \
              --num_processes=6 \
              --main_process_port 50008 \
              -m lmms_eval \
              --model qwen3_vl_ipcv \
              --model_args pretrained=$model_id,device_map=cuda,Sparse=$Sparse,vit_Sparse=$vit_Sparse,pruned_layer=$pruned_layer,vit_pruned_layer=$vit_pruned_layer,image_token_start_index=$image_token_start_index,image_token_length=$image_token_length,max_num_trunction=$max_num_trunction,reduction_ratio=$reduction_ratio,vit_reduction_ratio=$vit_reduction_ratio,pivot_image_token=$pivot_image_token,pivot_text_token=$pivot_text_token,random_choose=$random_choose,attn_scores_choose=$attn_scores_choose,diff_choose=$diff_choose,pivot_sim_choose=$pivot_sim_choose,vit_random_choose=$vit_random_choose,vit_attn_scores_choose=$vit_attn_scores_choose,vit_diff_choose=$vit_diff_choose,vit_pivot_sim_choose=$vit_pivot_sim_choose,torch_dtype=$torch_dtype,method=$method \
              --tasks "${task}" \
              --batch_size 1 \
              --log_samples \
              --output_path "$output_path"


          # ���Ӽ��ʱ�����˿ڳ�ͻ
          sleep 10
        done
      done
    done
  done
done
