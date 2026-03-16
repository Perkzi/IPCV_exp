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

# def find_windows_min_indices(tensor, sx, sy):
#     b, N = tensor.size()
#     n = int(math.sqrt(N))
#     tensor = tensor.view(b, n, n)
#     h_window = n // sy
#     w_window = n // sx
#     tensor = tensor[:, :h_window * sy, :w_window * sx]
#     tensor_reshaped = tensor.view(b, h_window, sy, w_window, sx)
#     tensor_reshaped = tensor_reshaped.permute(0, 1, 3, 2, 4).contiguous()
#     tensor_reshaped = tensor_reshaped.view(b, h_window, w_window, sy * sx)
#     _, min_indices = tensor_reshaped.min(dim=-1, keepdim=True)
#     return min_indices # 形状 (b, h_window, w_window, 1)  表示每个窗口里最小值的位置（在窗口内部的索引）

# def find_windows_min_indices(tensor, sx, sy,grid_thw):
#     b, N = tensor.size()
#     # 用于qwen2vl
#     t, h, w = grid_thw[0]   # 或者按 batch 取对应的 h, w
#     tensor = tensor.view(b, t, h, w)
#     h_window = h // sy
#     w_window = w // sx
#     tensor = tensor[:, :, :h_window * sy, :w_window * sx]
#     tensor_reshaped = tensor.view(b, t, h_window, sy, w_window, sx)
#     tensor_reshaped = tensor_reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
#     tensor_reshaped = tensor_reshaped.view(b, t, h_window, w_window, sy * sx)
#     _, min_indices = tensor_reshaped.min(dim=-1, keepdim=True)
#     return min_indices.squeeze(1)
def find_windows_min_indices(tensor):
    """
    不分块版本：直接在一维序列上排序，返回最小值索引。
    tensor: [batch_size, seq_len]
    输出: [batch_size, seq_len, 1] 排序后的索引
    """
    b, N = tensor.size()
    # 对每个 batch 的序列做排序，取最小值索引
    _, min_indices = tensor.min(dim=-1, keepdim=True)
    return min_indices



dst_score_global= None #torch.full((1, 576), float('1000000000'))

def Filter(
    att_score: torch.Tensor,#  (seq_length, seq_length)
    metric: torch.Tensor,   # 这里就是 key_states  # (bsz, seq_length, head_dim)
    reduction_factor: int,
    include_class_token: bool = False,
    grid_thw=None,
) -> Union[Tuple[torch.Tensor, ...], Tuple[Any, ...]]:
    # 决定哪些 token 冗余，哪些保留。
    total_tokens = att_score.shape[1]

    adjusted_reduction = reduction_factor
    if adjusted_reduction <= 0:
        return att_score, att_score

    global dst_score_global
    seq_len = att_score.shape[1]   # 或者 att_score.shape[1]
    if dst_score_global is None:
        dst_score_global = torch.full((1, seq_len), 65504.0, device=att_score.device,dtype=att_score.dtype )
        dst_score_global = dst_score_global.to(att_score.device)
        

    with torch.no_grad():
        att_score = att_score.unsqueeze(0)

        # --------- 改动部分：没有 CLS 的情况 ---------
        if False: #  include_class_token:
            # 原始逻辑：CLS 与 patch 的相似度
            cls_patch_similarity = att_score[:, 0, 1:]
        else:
            # 新逻辑：用 mean key 替代 CLS
            # metric: (bsz, seq_len, head_dim) head_dim=80
            mean_key = metric.mean(dim=1)  # (bsz, head_dim)
            #print("mean key",mean_key.unsqueeze(1).shape, metric.shape) # b,1,head_dim  b,s,head_dim
            cos_sim = F.cosine_similarity(mean_key.unsqueeze(1), metric, dim=-1)  # (bsz, seq_len)
            cls_patch_similarity = -cos_sim #.unsqueeze(0)  # ( bsz, seq_len)

        # ---------------------------------------------

        #patch_to_patch_similarity = att_score[..., 1:, 1:].mean(dim=-1)
        patch_to_patch_similarity = att_score.mean(dim=-1) # # (1,  seq_len)

        # 冗余度量
        beta = 0.35
        cls_patch_similarity = F.normalize(cls_patch_similarity, p=2, dim=-1)
        patch_to_patch_similarity = F.normalize(patch_to_patch_similarity, p=2, dim=-1)

        #print("cls_patch_similarity",cls_patch_similarity.shape, patch_to_patch_similarity.shape)
        dst_score = beta * cls_patch_similarity - (1 - beta) * patch_to_patch_similarity
        token_num = dst_score.shape[1]

        # 更新全局缓存
        remaining_indices = (dst_score_global != float('inf')).nonzero(as_tuple=True)[1]
        remaining_count = remaining_indices.numel()
        if remaining_count >= token_num:
            dst_score_global[0, remaining_indices[:token_num]] = dst_score
        else:
            dst_score_global[0, remaining_indices[:remaining_count]] = dst_score[0, :remaining_count]

        dst_score_pre = dst_score_global

        # 局部惩罚逻辑
        # sx, sy = 2, 2
        b, N = dst_score_pre.size()

        # 旧逻辑
        # n = int(math.sqrt(N))
        #hsy, wsx = n // sy, n // sx
        #rand_idx = find_windows_min_indices(dst_score_pre, sx, sy) # (b, hsy, wsx, 1)

        #---------- 用于qwen2vl 废弃------------
        # t, h, w = grid_thw[0]   # 或者按 batch 取对应的 h, w
        # # tensor = tensor.view(b, t, h, w)
        # hsy = h // sy
        # wsx = w // sx
        
        # rand_idx = find_windows_min_indices(dst_score_pre, sx, sy, grid_thw) # (b, hsy, wsx, 1)
        # idx_buffer_view = torch.zeros(b, hsy, wsx, sy * sx, device=dst_score.device, dtype=torch.int64)
        # idx_buffer_view.scatter_(dim=-1, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        # idx_buffer_view = idx_buffer_view.view(b, hsy, wsx, sy, sx).transpose(2, 3).reshape(b, hsy * sy, wsx * sx)
        # if (hsy * sy) < h or (wsx * sx) < w:
        #     idx_buffer = torch.zeros(b, h, w, device=dst_score.device, dtype=torch.int64)
        #     idx_buffer[:, :(hsy * sy), :(wsx * sx)] = idx_buffer_view
        # else:
        #     idx_buffer = idx_buffer_view
        #---------- 用于qwen2vl 废弃 end------------
        


        rand_idx = find_windows_min_indices(dst_score_pre)
        #print("rand_idx",rand_idx.shape)
        # rand_idx: [b, seq_len]，是排序后的索引

        # 构造 idx_buffer_view，直接在一维序列上 scatter
        idx_buffer_view = torch.zeros(b, token_num, device=dst_score.device, dtype=torch.int64)
        # 将排序结果里选中的位置标记为 -1
        idx_buffer_view.scatter_(dim=-1, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))

        # 不再需要 reshape 成二维网格，直接作为 idx_buffer 使用
        idx_buffer = idx_buffer_view



        rand_idx = idx_buffer.reshape(b, -1).argsort(dim=1)

        #rand_idx = rand_idx[:, :144]
        rand_idx = rand_idx[:, :token_num // 4]   # 修改：取 1/4 的 token

        rand_id_filtered = rand_idx[rand_idx < token_num]
        rand_id_filtered = rand_id_filtered.flatten()
        dst_score[0, rand_id_filtered] = torch.where(
            dst_score[0, rand_id_filtered] > 0,
            dst_score[0, rand_id_filtered] / 2,
            dst_score[0, rand_id_filtered] * 2
        )

        dst_non_zero_values = dst_score[dst_score != float('inf')]
        #large_value_tensor = torch.full((1,), float('inf'), device=dst_score.device, dtype=dst_score.dtype)
        #dst_non_zero_values = torch.cat((large_value_tensor, dst_non_zero_values), dim=0)
        dst_score_global[0, rand_id_filtered] = float('inf')

        order_indices = dst_non_zero_values.argsort(dim=-1, descending=False)[..., None].unsqueeze(0)
        merge_indices = order_indices[..., :reduction_factor, :]
        remain_indices = order_indices[..., reduction_factor:, :]

        # order_indices：所有 token 按冗余分数排序。
        # merge_indices：前 reduction_factor 个索引 → 要剪掉/合并的 token。
        # remain_indices：剩下的索引 → 保留的 token。
        # 所以最终剪掉的数量就是 reduction_factor

        return order_indices, merge_indices, remain_indices, att_score



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

    final_embeddings = unmerged_tokens
    return final_embeddings