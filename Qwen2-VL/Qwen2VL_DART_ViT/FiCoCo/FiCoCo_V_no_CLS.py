import torch
import torch.nn as nn

# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,CLIPConfig,CLIPVisionConfig
# from transformers.models.clip.modeling_clip import CLIPAttention ,CLIPMLP,CLIPEncoderLayer,CLIPVisionTransformer,CLIPEncoder
# from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
import math
from typing import Callable, Tuple,List
import torch.nn.functional as F

def merge_ficoco_v(
    Compress: Callable, 
    input_embeddings: torch.Tensor, 
    merge_indices: torch.Tensor,
    remain_indices: torch.Tensor,
    top_values:torch.Tensor,
    merge_targets: torch.Tensor,
    reduction_factor:int,
    size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if size is None:
        size = torch.ones_like(input_embeddings[..., 0, None])

    input_embeddings = Compress(
        input_embeddings * size,
        merge_indices,
        remain_indices,
        top_values,
        merge_targets,
        reduction_factor,
    )
    size = Compress(size, merge_indices, remain_indices, top_values,merge_targets, reduction_factor)

    input_embeddings = input_embeddings / size
    return input_embeddings, size



# 1. 适配变长分辨率的窗口索引查找
def find_windows_min_indices_qwen(tensor, sx, sy, h, w):
    """
    tensor: (1, total_tokens) 映射回原始 H*W 后的分数图
    h, w: 来自 grid_thw
    """
    b, N = tensor.size()
    # 还原为 2D
    tensor_2d = tensor.view(b, h, w)
    
    h_window = h // sy
    w_window = w // sx
    
    # 裁剪掉不能整除边缘（Qwen2-VL 转换时通常已对齐，这里做保险）
    tensor_2d = tensor_2d[:, :h_window * sy, :w_window * sx]
    
    # 划分为窗口 [b, h_w, sy, w_w, sx]
    tensor_reshaped = tensor_2d.view(b, h_window, sy, w_window, sx)
    # 重排为 [b, h_w, w_w, sy, sx] -> [b, h_w, w_w, sy*sx]
    tensor_reshaped = tensor_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    tensor_reshaped = tensor_reshaped.view(b, h_window, w_window, sy * sx)
    
    # 在每个窗口内找最小值索引 (0~3)
    _, min_indices = tensor_reshaped.min(dim=-1, keepdim=True)
    return min_indices, h_window, w_window

# 2. 全局变量（建议在模型 forward 开始前手动重置为 None）
dst_score_global = None 

def Filter(
    att_score: torch.Tensor,     # (seq_len, seq_len)
    metric: torch.Tensor,        # (bsz, seq_len, head_dim)
    reduction_factor: int,
    include_class_token: bool = False,
    grid_thw=None,               # [t, h, w]
):
    global dst_score_global
    
    # Qwen2-VL 动态获取当前图片的 H, W
    t, h, w = grid_thw[0]
    total_original_tokens = h * w
    current_token_num = att_score.shape[0]

    # 初始化或检查全局缓存
    if dst_score_global is None or dst_score_global.shape[1] != total_original_tokens or current_token_num == total_original_tokens:
        
        # 初次进入，全置为极大值 (1e9)
        dst_score_global = torch.full((1, total_original_tokens), 65504.0, 
                                     device=att_score.device, dtype=att_score.dtype)
        #print("initial dst_score_global",dst_score_global.shape)
    #print("dst_score_global",dst_score_global.shape)
    with torch.no_grad():
        # --- 步骤 1: 计算当前 Token 的冗余分数 ---
        # 针对无 CLS 模型：与均值向量相似度越高，分数越负，越容易被剪
        mean_key = metric.mean(dim=1) 
        cos_sim = F.cosine_similarity(mean_key.unsqueeze(1), metric, dim=-1) # (1, current_seq_len)
        cls_patch_similarity = -cos_sim 
        
        patch_to_patch_similarity = att_score.mean(dim=-1).unsqueeze(0) # (1, current_seq_len)
        
        # 归一化
        cls_p_sim = F.normalize(cls_patch_similarity, p=2, dim=-1)
        patch_p_sim = F.normalize(patch_to_patch_similarity, p=2, dim=-1)
        
        beta = 0.35
        # dst_score 越小越冗余
        dst_score = beta * cls_p_sim - (1 - beta) * patch_p_sim # (1, current_seq_len)

        # --- 步骤 2: 空间局部惩罚 (保护局部特征点) ---
        # 找出当前这些 Token 在原始 H*W 网格中的位置
        remaining_indices = (dst_score_global != float('inf')).nonzero(as_tuple=True)[1]
        
        # 构造一个临时全图，用于跑窗口查找
        temp_score_map = torch.full((1, total_original_tokens), 65504.0, 
                                   device=att_score.device, dtype=att_score.dtype)
        # 确保只在还有效的位置填入当前分数
        temp_score_map[0, remaining_indices] = dst_score[0]
        
        sx, sy = 2, 2
        # 在 2D 空间寻找窗口最小值索引
        win_min_idx, h_w, w_w = find_windows_min_indices_qwen(temp_score_map, sx, sy, h, w)
        
        # 构造保护掩码 (在全图 H*W 上)
        idx_buffer = torch.zeros(1, h_w, w_w, sy * sx, device=att_score.device, dtype=att_score.dtype)
        idx_buffer.scatter_(dim=-1, index=win_min_idx, src=torch.ones_like(win_min_idx, dtype=att_score.dtype))
        
        

        # ------------------------------v2-------------------------------
        idx_buffer = idx_buffer.view(1, h_w, w_w, sy, sx).transpose(2, 3).reshape(1, h_w * sy, w_w * sx)
        full_idx_map = torch.zeros(1, h, w, device=att_score.device, dtype=att_score.dtype)
        full_idx_map[:, :h_w*sy, :w_w*sx] = idx_buffer
        full_idx_map = full_idx_map.view(1, -1) # (1, total_original_tokens)
        # 【这里是需要加的逻辑】：复现原作者的 argsort 过滤
        # 原作者逻辑：在所有标记为“窗口最小值”的候选人里，按分数排序选前 N 个
        # 这里用一个小技巧：把非候选人的分数设得极大，然后排序
        tmp_for_sort = torch.full_like(temp_score_map, 65504.0)
        # 只在是“窗口最小值”的地方填入真实分数
        candidate_mask = (full_idx_map > 0)
        tmp_for_sort[candidate_mask] = temp_score_map[candidate_mask]
        
        # 排序拿到最该保护的索引
        sorted_indices = tmp_for_sort.argsort(dim=1, descending=False)
        
        # 原版是 144/576=0.25。Qwen2 按比例取：
        num_to_protect = total_original_tokens // 4 
        top_protected_indices = sorted_indices[:, :num_to_protect]
        
        # 构造最终的布尔掩码
        final_protect_mask_global = torch.zeros_like(full_idx_map, dtype=torch.bool)
        final_protect_mask_global.scatter_(1, top_protected_indices, True)

        # 映射回当前幸存者序列
        current_protected_mask = final_protect_mask_global[0, remaining_indices]
        # --------------------------------------------------------------------------
        # ------------v1----------------------
        # 还原回全图一维
        # idx_buffer = idx_buffer.view(1, h_w, w_w, sy, sx).transpose(2, 3).reshape(1, h_w * sy, w_w * sx)
        # full_mask = torch.zeros(1, h, w, device=att_score.device, dtype=att_score.dtype)
        # full_mask[:, :h_w*sy, :w_w*sx] = idx_buffer
        # full_mask = full_mask.view(1, -1).bool() # (1, total_original_tokens)
        
        # # 映射回当前序列：哪些是需要保护的 Token
        # current_protected_mask = full_mask[0, remaining_indices]
        # -----------------------------------
        
        # 对保护对象进行“提分”，让它不被 argsort 排在前面（即不被剪）
        # 分数越小越容易剪，所以这里我们通过增大正数或缩小负数来提分
        dst_score[0, current_protected_mask] = torch.where(
            dst_score[0, current_protected_mask] > 0,
            dst_score[0, current_protected_mask] * 2.0,
            dst_score[0, current_protected_mask] / 2.0
        )

        # --- 步骤 3: 排序并输出索引 ---
        # 我们只对“当前序列”进行排序
        # 注意：这里千万不能去掉 inf，直接排整个 dst_score
        order_indices_local = dst_score[0].argsort(dim=-1, descending=False)
        
        order_indices = order_indices_local.view(1, -1, 1)
        merge_indices = order_indices[:, :reduction_factor, :]
        remain_indices = order_indices[:, reduction_factor:, :]

        # --- 步骤 4: 更新全局状态 ---
        # 将本次被剪掉的 Token 在全局图中标记为 inf
        pruned_local_idx = order_indices_local[:reduction_factor]
        pruned_global_idx = remaining_indices[pruned_local_idx]
        dst_score_global[0, pruned_global_idx] = float('inf')

        return order_indices, merge_indices, remain_indices, att_score.unsqueeze(0)


def Correlate(
    merge_indices: torch.Tensor,
    att_scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入：merge_indices 和注意力分数 att_scores。
    逻辑：
    从要合并的 token 中，找出它们最相关的目标 token（通过注意力 top-k）。
    输出：
    target_indices：每个被合并 token对应的目标 token索引。
    top_values：这些相关性的权重。
    作用：确定“冗余 token 要合并到谁身上”。
    """
    merge_indices_choose = merge_indices.squeeze(-1)
    choose_att_prob = att_scores[0, merge_indices_choose[0], :]
    threshold_values = torch.quantile(choose_att_prob.float(), 0.998, dim=-1, keepdim=True)
    mask = choose_att_prob > threshold_values
    topk_per_token = mask.sum(dim=-1)
    max_topk = topk_per_token.max().item()
    att_prob = nn.functional.softmax(att_scores, dim=-1)
    top_values, top_indices = att_prob.topk(max_topk, dim=-1)
    expand_merge_indices = merge_indices.expand(-1, -1, top_indices.shape[-1])
    target_indices = top_indices.gather(dim=-2, index=expand_merge_indices)
    return target_indices, top_values

def Compress(
    input_embeddings:torch.Tensor,
    merge_indices: torch.Tensor,
    remain_indices: torch.Tensor,
    top_values:torch.Tensor,
    merge_targets: torch.Tensor,
    reduction_factor:int,
) -> torch.Tensor:
    # 作用：真正执行 token 合并，得到压缩后的 hidden states。
    #print("input_embeddings",input_embeddings,input_embeddings.shape)
    merge=True
    num_samples, num_pairs, num_features = input_embeddings.shape
    unmerged_tokens = input_embeddings.gather(dim=-2, index=remain_indices.expand(num_samples, num_pairs - reduction_factor, num_features))
    source_tokens = input_embeddings.gather(dim=-2, index=merge_indices.expand(num_samples, reduction_factor, num_features))
    if merge_targets.shape[-1]!=0:
        total_scores = top_values.sum(dim=-1, keepdim=True)
        merge_weights = top_values / total_scores
        merge_indices_choose = merge_indices.squeeze(-1)
        chosen_weights = merge_weights.gather(dim=1, index=merge_indices_choose.unsqueeze(-1).expand(-1, -1, merge_weights.size(-1)))
        weighted_tokens = source_tokens.unsqueeze(2) * chosen_weights.unsqueeze(-1)
        expanded_tokens = weighted_tokens.reshape(num_samples, -1, num_features)
        flat_indices = merge_targets.reshape(num_samples, -1).unsqueeze(-1).expand(-1, -1, num_features)
        max_length = unmerged_tokens.shape[1]
        flat_indices = torch.where(flat_indices >= max_length, flat_indices - reduction_factor, flat_indices)
        unmerged_tokens.scatter_add_(1, flat_indices, expanded_tokens)

    #final_embeddings = unmerged_tokens

    # --- 新增：在这里将顺序重排回正序 ---
    # 1. 对 remain_indices 进行排序，拿到“如何变回正序”的映射索引 restore_idx
    # remain_indices 形状通常是 (1, num_remain, 1)
    _, restore_idx = torch.sort(remain_indices.squeeze(-1), dim=-1) # (num_samples, num_remain)
    
    # 2. 按照 restore_idx 重新排列 unmerged_tokens
    final_embeddings = unmerged_tokens.gather(
        dim=1, 
        index=restore_idx.unsqueeze(-1).expand(-1, -1, num_features)
    )
    return final_embeddings