# DivPrune (diversity-based visual token pruning) for Qwen2-VL ViT.
# Algorithm aligned with vbdi/divprune LLaVA `DivPrune` in llava_arch.py.
# Retained token count uses the same convention as `modeling_qwen2_vl_base.DART_ViT`:
# keep ratio ≈ 1 - vit_reduction_ratio (see `vit_reduction_ratio` in config.DART_config).

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_qwen2_vl_base import (
    DART,
    DART_ViT,
    Qwen2VLForConditionalGeneration as Qwen2VLFCBase,
    Qwen2VLPreTrainedModel,
)


def _pairwise_cosine_similarity(matrix: torch.Tensor) -> torch.Tensor:
    denom = matrix.norm(dim=1, keepdim=True).clamp(min=1e-6)
    norm_matrix = matrix / denom
    return torch.mm(norm_matrix, norm_matrix.t())


def _divprune_select_indices(visual_feature_vectors: torch.Tensor, keep_count: int) -> torch.Tensor:
    """
    Greedy diversity selection; returns `keep_count` distinct token indices (unordered).
    Matches divprune `DivPrune` when `threshold_ratio * N == keep_count`.
    """
    n = visual_feature_vectors.shape[0]
    device = visual_feature_vectors.device
    keep_count = max(1, min(int(keep_count), n))
    if keep_count >= n:
        return torch.arange(n, device=device, dtype=torch.long)

    cosine_matrix = 1.0 - _pairwise_cosine_similarity(visual_feature_vectors)

    selected = torch.empty(keep_count, dtype=torch.long, device=device)
    for i in range(keep_count):
        if i == 0:
            m2 = cosine_matrix
            # second-smallest per column (see original DivPrune)
            scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]
        else:
            m2 = torch.index_select(
                cosine_matrix, 0, torch.index_select(selected, 0, torch.arange(0, i, device=device))
            )
            scores = torch.min(m2, dim=0).values
        selected[i] = torch.argmax(scores)

    return selected


def _vit_keep_count_aligned_4(image_token_length: int, vit_reduction_ratio: float) -> int:
    """Same rounding as `get_retained_image_token_random` in base `DART_ViT`."""
    reduction_ratio = float(vit_reduction_ratio)
    token_topk_raw = image_token_length * (1.0 - reduction_ratio)
    token_topk_down = int(token_topk_raw) // 4 * 4
    token_topk_up = (int(token_topk_raw) + 3) // 4 * 4
    if abs(token_topk_raw - token_topk_down) <= abs(token_topk_raw - token_topk_up):
        retained_count = token_topk_down
    else:
        retained_count = token_topk_up
    retained_count = max(retained_count, 1)
    return min(retained_count, image_token_length)


class DivPruneDART_ViT(DART_ViT):
    """`DART_ViT` with ViT-stage DivPrune token selection at `vit_pruned_layer`."""

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        device = hidden_states.device
        dtype = hidden_states.dtype

        if (
            self.config.DART_config is not None
            and self.config.DART_config["vit_Sparse"]
            and self.config.DART_config["vit_attn_scores_choose"]
            and not self.update_attention_layer
        ):
            self.update_vision_block(device, dtype)
            self.update_attention_layer = True

        hidden_states_pkg = {
            "hidden_states": hidden_states,
            "k_states": None,
            "attn_scores": None,
        }
        frame_counts = torch.zeros(1, device=device)

        cu_seqlens_pruned = None
        rotary_pos_emb_pruned = None

        for blk in self.blocks:
            DART_config = self.config.DART_config
            if DART_config is not None and DART_config["vit_Sparse"]:
                K = DART_config["vit_pruned_layer"]

                if blk.layer_idx == K and hidden_states_pkg["hidden_states"].shape[0] > 1:
                    device = hidden_states_pkg["hidden_states"].device
                    last_layer_state = hidden_states_pkg["hidden_states"].detach().clone()
                    last_layer_state = self.norm(last_layer_state)

                    image_token_length = last_layer_state.shape[0]
                    keep_n = _vit_keep_count_aligned_4(image_token_length, DART_config["vit_reduction_ratio"])
                    retained = _divprune_select_indices(last_layer_state, keep_n)
                    keep_indexs = retained.sort().values

                    orig_seq_len = hidden_states_pkg["hidden_states"].shape[0]
                    orig_states = hidden_states_pkg["hidden_states"].detach().clone()

                    hidden_states_pkg["hidden_states"] = orig_states[keep_indexs, :]
                    rotary_pos_emb_pruned = rotary_pos_emb[keep_indexs, :]

                    self._sparse_vit_saved = {"orig_seq_len": orig_seq_len}

                    num_frames = len(cu_seqlens) - 1
                    frame_counts = torch.zeros(num_frames, dtype=torch.int32, device=device)
                    frame_indices = torch.searchsorted(cu_seqlens, keep_indexs, right=False) - 1
                    frame_indices = frame_indices.clamp(min=0, max=num_frames - 1)
                    frame_counts = torch.bincount(frame_indices, minlength=num_frames)

                    new_cu_seqlens = torch.zeros(len(cu_seqlens), dtype=torch.int32, device=device)
                    new_cu_seqlens[0] = 0
                    new_cu_seqlens[1:] = frame_counts.cumsum(dim=0)
                    cu_seqlens_pruned = new_cu_seqlens.to(torch.int32)

            if hasattr(self, "_sparse_vit_saved"):
                hidden_states_pkg = blk(
                    hidden_states_pkg,
                    cu_seqlens=cu_seqlens_pruned,
                    rotary_pos_emb=rotary_pos_emb_pruned,
                )
            else:
                hidden_states_pkg = blk(
                    hidden_states_pkg, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
                )

            if blk.layer_idx == 32 - 1 and hasattr(self, "_sparse_vit_saved"):
                del self._sparse_vit_saved

        frame_counts = frame_counts / (self.config.spatial_merge_size**2)

        return self.merger(hidden_states_pkg["hidden_states"]), frame_counts


class Qwen2VLForConditionalGeneration(Qwen2VLFCBase):
    """Same as base but ViT uses DivPrune selection (`DivPruneDART_ViT`)."""

    def __init__(self, config):
        Qwen2VLPreTrainedModel.__init__(self, config)
        self.visual = DivPruneDART_ViT._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.model = DART(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"
        self.post_init()
        self.time_cost_vit = 0
        self.time_cost_llm = 0


__all__ = ["DivPruneDART_ViT", "Qwen2VLForConditionalGeneration"]
