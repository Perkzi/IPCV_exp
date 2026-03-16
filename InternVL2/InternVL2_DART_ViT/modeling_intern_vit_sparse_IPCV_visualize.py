# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.layers import DropPath
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import \
        flash_attn_varlen_qkvpacked_func
    has_flash_attn = True
except:
    print('FlashAttention2 is not installed.')
    has_flash_attn = False

import math

logger = logging.get_logger(__name__)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                          device=qkv.device)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                             indices, batch_size, seqlen),
                                   'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm')
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning('discovered apex but it failed to load, falling back to InternRMSNorm')
    pass


NORM2FN = {
    'rms_norm': InternRMSNorm,
    'layer_norm': nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, k  

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        q, k, v = qkv.unbind(2)
        if self.qk_normalization:
            
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs, None, k

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x, attn ,k = self._naive_attn(hidden_states) if not self.use_flash_attn else self._flash_attn(hidden_states)
        return {'hidden_states':x,
                'attn_scores':attn,
                'k_states':k}


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(
            self,
            hidden_states: torch.Tensor,
            sparse_vit_saved=None
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        if sparse_vit_saved is not None:
            saved = sparse_vit_saved  # 或 sparse_vit_saved
            B, L, D = saved["removed_states"].size(0), saved["orig_seq_len"], saved["removed_states"].size(2)
            device = hidden_states.device  # 或 hidden_states.device

            # 全局 new_kept / orig_kept
            K = saved["orig_kept_states"].shape[1]
            flat_new_kept = hidden_states.reshape(B*K, D)  # 或 hidden_states.reshape(B*K, D)
            flat_orig_kept = saved["orig_kept_states"].reshape(B*K, D)

            full_states_patch = []
            for b in range(B):
                rem_to_kept_idx = saved["rem_to_kept_idx"][b]  # [R, topk]，全局索引
                removed_indexs_per_batch = saved["removed_indices"][b]
                unique_idx = saved["unique_idx"][b]
                inv_idx = saved["inv_idx"][b]

                # 计算 delta（全局索引）
                delta_unique = flat_new_kept[unique_idx] - flat_orig_kept[unique_idx]  # [U, D]

                inv_idx = inv_idx.view(rem_to_kept_idx.shape)  # [R, topk]
                avg_delta_removed = delta_unique[inv_idx].mean(dim=1)  # [R, D]

                # 准备 full_states
                full_states = torch.zeros(L, D, device=device, dtype=flat_new_kept.dtype)
                # 当前 batch 的 kept 索引还是局部的
                full_states[saved["keep_indexs"][b]] = hidden_states[b]  # 或 hidden_states[b]
                full_states[removed_indexs_per_batch] = saved["removed_states"][b] + avg_delta_removed

                full_states_patch.append(full_states)

            full_states_patch = torch.stack(full_states_patch, dim=0)
            hidden_states = full_states_patch  # 或 hidden_states = full_states_patch


            # # 加上变化量并拼接
            # saved = sparse_vit_saved
            # B, L, D = saved["removed_states"].size(0), saved["orig_seq_len"], saved["removed_states"].size(2)
            # device = hidden_states.device

            # full_states_patch = []
            # for b in range(B):
            #     # 1) 拿出新旧 kept_states 与 rem_to_kept_idx
            #     new_kept        = hidden_states[b]       # [K, D]
            #     orig_kept       = saved["orig_kept_states"][b]               # [K, D]
            #     rem_to_kept_idx = saved["rem_to_kept_idx"][b]                # [R, 10]
            #     removed_indexs_per_batch = saved["removed_indices"][b]                # [R]
            #     unique_idx = saved["unique_idx"][b]
            #     inv_idx = saved["inv_idx"][b]

            #     # 2) 只对 unique_idx 计算一次 delta
            #     delta_unique = (new_kept[unique_idx] - orig_kept[unique_idx])              # [U, D]

            #     # 3) 把 inv_idx reshape 回 (R, topk)，再 gather 并均值
            #     inv_idx = inv_idx.view(rem_to_kept_idx.shape)                              # [R, topk]
            #     #print("delta_unique",delta_unique[inv_idx].shape)
            #     avg_delta_removed = delta_unique[inv_idx].mean(dim=1)                      # [R, D]

            #     # 4) 准备 full_states 并写回
            #     full_states = torch.zeros(L, D, device=device, dtype=new_kept.dtype)
            #     #print("fullstates",hidden_states.shape,full_states.shape,saved["keep_indexs"].shape,saved["keep_indexs"])
                
            #     full_states[saved["keep_indexs"][b]] = new_kept                                # fill kept
            #     full_states[removed_indexs_per_batch]    = (saved["removed_states"][b] + avg_delta_removed)

            #     full_states_patch.append(full_states)
            # full_states_patch = torch.stack(full_states_patch,dim=0)

            # # 5) 替换并清理
            # hidden_states = full_states_patch

        attn_outputs = self.attn(self.norm1(hidden_states).to(hidden_states.dtype))
        #print("attn_out",attn_outputs)
        hidden_states = hidden_states + self.drop_path1(attn_outputs['hidden_states'] * self.ls1)

        if sparse_vit_saved is not None:
            # 重新裁剪
            pruned = []
            for b in range(B):
                pruned.append(hidden_states[b, saved["keep_indexs"][b], :])  # (K, C)
            hidden_states = torch.stack(pruned, dim=0)  # (B, K, C)

        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2)

        #return hidden_states
    
        return {'hidden_states':hidden_states,
                'attn_scores':attn_outputs['attn_scores'],
                'k_states':attn_outputs['k_states']}


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


class InternVisionEncoder_Sparse(InternVisionEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)

        self.update_attention_layer=False

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds



        device = hidden_states.device
        dtype = hidden_states.dtype
        
        if self.config.DART_config is not None and self.config.DART_config['vit_Sparse'] and self.config.DART_config['vit_attn_scores_choose']\
            and not self.update_attention_layer:
            k = self.config.DART_config['vit_pruned_layer'] - 1
            self.layers[k].attn.use_flash_attn = False
            self.update_attention_layer=True

        #--------------------BEGIN------------------------------------
        hidden_states_pkg = {'hidden_states':hidden_states, # [batch_size,seq_len, embed_dim]
                            'k_states':None,                # [batch_size,seq_len, num_heads, head_dim]  TODO:优化显存占用
                            'attn_scores':None}                 # [batch_size, nheads,seqlen,seqlen] TODO: 优化显存占用
        #frame_counts = torch.zeros(1, device=device)
        hidden_states_prev = None
        #print("hid state",hidden_states.shape,len(self.layers))

        
        # self.analyzer = TokenTrajectoryAnalyzer()
        
        if not hasattr(self, 'consistency_analyzer'):
            self.consistency_analyzer = EvolutionConsistencyAnalyzer()

        for idx, blk in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states_pkg = torch.utils.checkpoint.checkpoint(
                    blk,
                    hidden_states
                )
                hidden_states = hidden_states_pkg['hidden_states']
            else:
                DART_config = self.config.DART_config
                if DART_config is not None and DART_config['vit_Sparse']:
                    K = DART_config['vit_pruned_layer']
                    seq_len = hidden_states_pkg['hidden_states'].shape[1]  # (B, N, C) 取 N

                    if K - 1 > 0 and idx == K - 1 and DART_config['vit_diff_choose'] and seq_len > 1:
                        hidden_states_prev = hidden_states_pkg['hidden_states']  # (B, N, C)

                    if idx == K and seq_len > 1:
                        device = hidden_states_pkg['hidden_states'].device
                        last_layer_state = hidden_states_pkg['hidden_states'].detach().clone()  # (B, N, C)
                        k_states = hidden_states_pkg['k_states']   # (B, heads, N, C) 或类似
                        attn_scores = hidden_states_pkg['attn_scores']  # (B, heads, N, N) 或类似
                        #print("k_states",k_states.shape)

                        orig_seq_len = seq_len

                        B = last_layer_state.shape[0]
                        keep_indexs_per_batch = []
                        removed_indexs_per_batch = []

                        for b in range(B):
                            if DART_config['vit_attn_scores_choose']:
                                keep_idx = self.get_retained_image_token_attn_scores(
                                    self.config,
                                    last_layer_state[b,1:],  # 单样本
                                    k_states[b,1:],
                                    attn_scores[b,1:]
                                ).to(device)
                            elif DART_config['vit_random_choose']:
                                keep_idx = self.get_retained_image_token_random(
                                    self.config,
                                    last_layer_state[b,1:],
                                    k_states[b,1:]
                                ).to(device)
                            elif DART_config['vit_diff_choose']:
                                keep_idx = self.get_retained_image_token_diff(
                                    self.config,
                                    hidden_states_pkg['hidden_states'][b,1:],
                                    hidden_states_prev[b,1:],
                                    last_layer_state[b,1:]
                                )
                            elif DART_config['vit_pivot_sim_choose']:
                                keep_idx = self.get_retained_image_token_pivot_sim(
                                    self.config,
                                    last_layer_state[b,1:],
                                    k_states[b,1:]
                                )
                            else:
                                keep_idx = self.get_retained_image_token(
                                    self.config,
                                    last_layer_state[b,1:],
                                    k_states[b,1:]
                                ).to(device)

                            keep_idx = keep_idx.sort().values
                           
                            # 所有索引 +1（因为 CLS 在第 0 位）
                            keep_idx = keep_idx + 1

                            # 在最前面加 CLS 的索引 0
                            keep_idx = torch.cat([
                                torch.tensor([0], device=keep_idx.device, dtype=keep_idx.dtype),
                                keep_idx
                            ])

                            removed_mask = torch.ones(orig_seq_len, dtype=torch.bool, device=device)
                            removed_mask[keep_idx] = False
                            removed_idx = torch.nonzero(removed_mask, as_tuple=False).view(-1)

                           
                            keep_indexs_per_batch.append(keep_idx)
                            removed_indexs_per_batch.append(removed_idx)
                        keep_indexs_per_batch = torch.stack(keep_indexs_per_batch, dim=0)
                        removed_indexs_per_batch = torch.stack(removed_indexs_per_batch, dim=0)

                        # 保存原始长度
                        orig_states = hidden_states_pkg['hidden_states'].detach().clone()

                        # 对每个样本单独裁剪
                        orig_kept_states = []
                        removed_states = []
                        for b in range(B):
                            orig_kept_states.append(orig_states[b, keep_indexs_per_batch[b], :])
                            removed_states.append(orig_states[b, removed_indexs_per_batch[b], :])
                        orig_kept_states = torch.stack(orig_kept_states, dim=0)
                        removed_states = torch.stack(removed_states, dim=0)


                        with torch.no_grad():
                            # 拼成全局 kept token
                            B, K, D = orig_kept_states.shape
                            orig_kept_states_all = orig_kept_states.reshape(B*K, D)  # (B*K, D)

                            rem_to_kept_idx_patch = []
                            unique_idx_patch = []
                            inv_idx_patch = []

                            for b in range(B):
                                # 计算 removed_states[b] 到全局 kept token 的距离
                                dists = torch.cdist(
                                    removed_states[b].float(),
                                    orig_kept_states_all.float(),
                                    p=2.0
                                )
                                # topk 最小距离对应的全局 kept 索引
                                _, rem_to_kept_idx_global = dists.topk(min(10, orig_kept_states_all.shape[0]), largest=False, dim=1)

                                flat_idx = rem_to_kept_idx_global.view(-1)
                                unique_idx, inv_idx = torch.unique(flat_idx, return_inverse=True)

                                rem_to_kept_idx_patch.append(rem_to_kept_idx_global)
                                unique_idx_patch.append(unique_idx)
                                inv_idx_patch.append(inv_idx)



                                # 假设你在这个位置：
                                # dists = torch.cdist(removed_states[b].float(), orig_kept_states_all.float(), p=2.0)
                                # 我们分析第 0 个 patch 的某个 token（或者随机选一个 patch）
                                
                                # 只需要在你的 for b in range(B) 循环里加这一句：
                                if hasattr(self, 'analyzer'):
                                    if b == 0: # 仅分析第一个 patch 里的删除情况
                                        self.analyzer.select_targets(
                                            removed_indexs_per_batch[0], 
                                            keep_indexs_per_batch, 
                                            rem_to_kept_idx_global
                                        )

                            rem_to_kept_idx_patch = torch.stack(rem_to_kept_idx_patch, dim=0)
                        # with torch.no_grad():
                        #     # p=2.0             ：指定用 L2 范数（Euclidean，p=2）；如果 p=1 则是 L1 距离，p=∞ 则是 Chebyshev 距离，等等
                        #     # 输出 dists       ：shape=[R, K]，其中 dists[i,j] 是 removed_states[i] 和 orig_kept_states[j] 的 p‐范数距离
                        #     # pairwise distance: [R, K]
                        #     #dists = torch.cdist(removed_states, orig_kept_states, p=2.0)
                        #     #print(removed_states.shape,orig_kept_states.shape)
                        #     rem_to_kept_idx_patch = []
                        #     unique_idx_patch = []
                        #     inv_idx_patch = []
                        #     for b in range(B):
                        #         dists = torch.cdist(
                        #             removed_states[b].float(), 
                        #             orig_kept_states[b].float(), 
                        #             p=2.0
                        #         )
                        #         # topk 最小距离对应的 kept_states 索引： [R, 10]
                        #         _, rem_to_kept_idx = dists.topk(min(10,orig_kept_states[b].shape[0]), largest=False, dim=1)

                        #         flat_idx = rem_to_kept_idx.view(-1)                                        # [R*topk]
                        #         unique_idx, inv_idx = torch.unique(flat_idx, return_inverse=True)          # unique_idx:[U], inv_idx:[R*topk]
                                
                        #         rem_to_kept_idx_patch.append(rem_to_kept_idx)
                        #         unique_idx_patch.append(unique_idx)
                        #         inv_idx_patch.append(inv_idx)
                        #     rem_to_kept_idx_patch = torch.stack(rem_to_kept_idx_patch,dim=0)
                        #     #unique_idx_patch = torch.stack(unique_idx_patch,dim=0)
                        #     #inv_idx_patch = torch.stack(inv_idx_patch,dim=0)





                        # hidden_states_pkg['hidden_states'] = orig_kept_states # 不剪枝
                        

                        hidden_states = hidden_states_pkg['hidden_states']


                        self._sparse_vit_saved = {
                            "orig_seq_len": orig_seq_len,

                            "keep_indexs": keep_indexs_per_batch,
                            "removed_indices": removed_indexs_per_batch,
                            "removed_states": removed_states,
                            "orig_kept_states": orig_kept_states,

                            "orig_kept_states_flat": orig_kept_states.reshape(-1, D),

                            # 新增这行，R×10 的 LongTensor
                            "rem_to_kept_idx":    rem_to_kept_idx_patch,  
                            "unique_idx": unique_idx_patch,
                            "inv_idx": inv_idx_patch,
                        }

                        # --- 在 for b in range(B) 循环外调用 ---
                        # 假设你在第 K 层刚做完剪枝逻辑
                        if hasattr(self, 'consistency_analyzer'):
                            # 将列表转换为张量 [B, R, 10]
                            all_neighbors_tensor = rem_to_kept_idx_patch
                            
                            self.consistency_analyzer.record_k_layer(
                                hidden_states,          # [B, N, C]
                                removed_indexs_per_batch, # [B, R]
                                keep_indexs_per_batch,       # [B, K]
                                all_neighbors_tensor 
                            )

                    
                        
                    # 收集数据：从第 K 层到最后一层
                    if hasattr(self, 'analyzer'):
                        if idx >= K:
                            # 这里传入 hidden_states (确保它是 (B, N, C) 的原始完整状态)
                            self.analyzer.collect(idx, hidden_states)
                #print("hid state",hidden_states.shape)
                # if hasattr(self, "_sparse_vit_saved") and idx < DART_config['vit_pruned_layer']+3:
                #     hidden_states_pkg = blk(hidden_states, sparse_vit_saved=self._sparse_vit_saved)
                # else:
                hidden_states_pkg = blk(hidden_states)


                # if idx == self.config.num_hidden_layers - 1 and hasattr(self, "_sparse_vit_saved"):
                #     saved = self._sparse_vit_saved  # 或 sparse_vit_saved
                #     B, L, D = saved["removed_states"].size(0), saved["orig_seq_len"], saved["removed_states"].size(2)
                #     device = hidden_states_pkg['hidden_states'].device  # 或 hidden_states.device

                #     # 全局 new_kept / orig_kept
                #     K = saved["orig_kept_states"].shape[1]
                #     flat_new_kept = hidden_states_pkg['hidden_states'].reshape(B*K, D)  # 或 hidden_states.reshape(B*K, D)
                #     flat_orig_kept = saved["orig_kept_states"].reshape(B*K, D)

                #     full_states_patch = []
                #     for b in range(B):
                #         rem_to_kept_idx = saved["rem_to_kept_idx"][b]  # [R, topk]，全局索引
                #         removed_indexs_per_batch = saved["removed_indices"][b]
                #         unique_idx = saved["unique_idx"][b]
                #         inv_idx = saved["inv_idx"][b]

                #         # 计算 delta（全局索引）
                #         delta_unique = flat_new_kept[unique_idx] - flat_orig_kept[unique_idx]  # [U, D]

                #         inv_idx = inv_idx.view(rem_to_kept_idx.shape)  # [R, topk]
                #         avg_delta_removed = delta_unique[inv_idx].mean(dim=1)  # [R, D]

                #         # 准备 full_states
                #         full_states = torch.zeros(L, D, device=device, dtype=flat_new_kept.dtype)
                #         # 当前 batch 的 kept 索引还是局部的
                #         full_states[saved["keep_indexs"][b]] = hidden_states_pkg['hidden_states'][b]  # 或 hidden_states[b]
                #         full_states[removed_indexs_per_batch] = saved["removed_states"][b] + avg_delta_removed

                #         full_states_patch.append(full_states)

                #     full_states_patch = torch.stack(full_states_patch, dim=0)
                #     hidden_states_pkg['hidden_states'] = full_states_patch  # 或 hidden_states = full_states_patch

                #     del self._sparse_vit_saved

                hidden_states = hidden_states_pkg['hidden_states']

                if idx == self.config.num_hidden_layers - 1 and hasattr(self, 'consistency_analyzer'):
                    self.consistency_analyzer.record_final_layer(hidden_states)

        # --- 整个 Encoder 结束前保存 ---
        if hasattr(self, 'analyzer'):
            self.analyzer.finalize_and_save()
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


    
                    
                    

    
    def get_retained_image_token(self, config, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        # any_state [seq_len, num_heads, head_dim]
        DART_config = config.DART_config
        #K = DART_config['vit_pruned_layer']
        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]

        pivot_image_token = DART_config['pivot_image_token']
        # pivot_text_token = DART_config['pivot_text_token']

        reduction_ratio = DART_config['vit_reduction_ratio']
        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio) / (pivot_image_token)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     TOKEN_TOPK = TOKEN_TOPK_down
        # else:
        #     TOKEN_TOPK = TOKEN_TOPK_up

        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio) / (pivot_image_token)

        # # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        # import math
        # max_square_root = int(math.sqrt(pivot_image_token*TOKEN_TOPK_RAW))  # 向下取整平方根
        # TOKEN_TOPK = max_square_root ** 2  # 得到平方数

        # 原始计算
        TOKEN_TOPK = int(image_token_length * (1 - reduction_ratio) / pivot_image_token)

        # # 最大可能的平方根
        # max_square_root = int(math.sqrt(pivot_image_token * (TOKEN_TOPK_RAW+1)))

        # # 从 max_square_root 往下找，直到找到既是平方数又是 pivot_image_token 的倍数
        # TOKEN_TOPK = 0
        # for r in range(max_square_root, 0, -1):
        #     candidate = r ** 2
        #     if candidate % pivot_image_token == 0:
        #         TOKEN_TOPK = int(candidate /pivot_image_token - 1)
        #         break
        #print("topk",TOKEN_TOPK)


        # 向下取
        # TOKEN_TOPK = TOKEN_TOPK_down - 1
        device = last_layer_state.device

        device = last_layer_state.device

        any_states = any_states.reshape(any_states.shape[0], -1) # [seq_len, embed_dim]

        k_states_image_token = any_states[image_token_start_index:image_token_start_index + image_token_length, :] # [valid_seq_len, hidden_dim]
        #k_states_query_token = any_states[image_token_start_index + image_token_length:, :]

        k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1) # [valid_seq_len]
        #k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1) # [valid_seq_len]

        image_indices = (k_states_image_token_L1_norm.topk(pivot_image_token).indices + image_token_start_index).tolist() # pivot indices (list)
        #query_indices = (k_states_query_token_L1_norm.topk(pivot_text_token).indices + image_token_start_index + image_token_length).tolist() # pivot indices (list)
        #indices_set = set(image_indices + query_indices) # merge 2 lists
        indices_set = set(image_indices)

        valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(image_indices)

        valid_indices_list = list(valid_indices)
        for item in list(indices_set):
            valid_vectors = last_layer_state[valid_indices_list, :] # last_layer_state中待处理image token的对应向量 [valid_seq_len - num_pivot_tokens, hidden_dim]
            cos_sim = -torch.nn.functional.cosine_similarity(last_layer_state[item, :], valid_vectors, dim=-1) # 计算余弦相似度 [valid_seq_len - num_pivot_tokens]
            #print("cossim",cos_sim.shape)
            top_k_indices = cos_sim.topk(TOKEN_TOPK).indices

            top_k_real_indices = [valid_indices_list[i] for i in top_k_indices] # 待保留的image token的index
            indices_set.update(top_k_real_indices)

            valid_indices.difference_update(top_k_real_indices)
            valid_indices_list = list(valid_indices) 

            retained_image_tokens_index = torch.tensor(list(indices_set), device=device)

        return retained_image_tokens_index

    def get_retained_image_token_random(self, config, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        reduction_ratio = DART_config['vit_reduction_ratio']

        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]
        device = last_layer_state.device

        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     retained_count = TOKEN_TOPK_down
        # else:
        #     retained_count = TOKEN_TOPK_up

        # 计算原始值
        TOKEN_TOPK_RAW = int(image_token_length * (1 - reduction_ratio))

        # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        
        # max_square_root = int(math.sqrt(TOKEN_TOPK_RAW))  # 向下取整平方根
        # retained_count = max_square_root ** 2  # 得到平方数

        # 向下取
        # retained_count = TOKEN_TOPK_down
        # 确保至少保留一个token
        retained_count = max(TOKEN_TOPK_RAW, 1)
        
        # 生成所有图像token的索引并随机选择
        all_indices = torch.arange(image_token_start_index, image_token_start_index + image_token_length, device=device)
        retained_indices = all_indices[torch.randperm(all_indices.size(0))[:retained_count]]
        
        return retained_indices

    def get_retained_image_token_attn_scores(self, config, last_layer_state: torch.Tensor, any_states: torch.Tensor, attn_scores: torch.Tensor) -> torch.Tensor:
        # any_state [seq_len, num_heads, head_dim]
        DART_config = config.DART_config
        # K = DART_config['K']
        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]

        reduction_ratio = DART_config['vit_reduction_ratio']
        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     TOKEN_TOPK = TOKEN_TOPK_down
        # else:
        #     TOKEN_TOPK = TOKEN_TOPK_up

        # 计算原始值
        TOKEN_TOPK = int(image_token_length * (1 - reduction_ratio))

        # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        
        # max_square_root = int(math.sqrt(TOKEN_TOPK_RAW))  # 向下取整平方根
        # TOKEN_TOPK = max_square_root ** 2  # 得到平方数
        device = last_layer_state.device

        #attn_scores.squeeze(0) # [nheads,seqlen,seqlen]
        attn_scores = attn_scores.sum(dim=-2) # 沿着query维度求和
        attn_scores = attn_scores.mean(dim=0) # 对不同的注意力头求平均
        top_k_indices = attn_scores.topk(TOKEN_TOPK).indices
        top_k_real_indices = top_k_indices
        retained_image_tokens_index = torch.tensor(top_k_real_indices, device=device)
        return retained_image_tokens_index

    def get_retained_image_token_diff(self,config,hidden_states_prev,hidden_states_cur,last_layer_state):
        DART_config = config.DART_config
        #K = DART_config['K']
        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]

        reduction_ratio = DART_config['vit_reduction_ratio']
        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     TOKEN_TOPK = TOKEN_TOPK_down
        # else:
        #     TOKEN_TOPK = TOKEN_TOPK_up

        # 计算原始值
        TOKEN_TOPK = int(image_token_length * (1 - reduction_ratio))
        

        # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        
        # max_square_root = int(math.sqrt(TOKEN_TOPK_RAW))  # 向下取整平方根
        # TOKEN_TOPK = max_square_root ** 2  # 得到平方数

        #print("topk",TOKEN_TOPK_RAW,TOKEN_TOPK)
        # # 向下取
        # TOKEN_TOPK = TOKEN_TOPK_down - 1
        device = last_layer_state.device

        diff = hidden_states_cur - hidden_states_prev # [seqlen,embed_dim]
        diff_norm = torch.norm(diff,dim=-1) # [seqlen]
        top_k_indices = diff_norm.topk(TOKEN_TOPK).indices
        top_k_real_indices = top_k_indices
        retained_image_tokens_index = torch.tensor(top_k_real_indices, device=device)
        return retained_image_tokens_index

    def get_retained_image_token_pivot_sim(self, config, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        #K = DART_config['K']
        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]

        reduction_ratio = DART_config['vit_reduction_ratio']
        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     TOKEN_TOPK = TOKEN_TOPK_down
        # else:
        #     TOKEN_TOPK = TOKEN_TOPK_up

        # 计算原始值
        TOKEN_TOPK = int(image_token_length * (1 - reduction_ratio))

        # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        
        # max_square_root = int(math.sqrt(TOKEN_TOPK_RAW))  # 向下取整平方根
        # TOKEN_TOPK = max_square_root ** 2  # 得到平方数
        
        device = last_layer_state.device
        pivot_token = last_layer_state.mean(dim=0) # 求出平均token [embed_dim]
        cos_sim = -torch.nn.functional.cosine_similarity(pivot_token, last_layer_state, dim=-1) # 计算余弦相似度
        top_k_indices = cos_sim.topk(TOKEN_TOPK).indices
        top_k_real_indices = top_k_indices
        retained_image_tokens_index = torch.tensor(top_k_real_indices, device=device)
        return retained_image_tokens_index

# import torch
# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA

# class TokenTrajectoryAnalyzer:
#     def __init__(self, target_layer_start):
#         self.start_layer = target_layer_start
#         self.target_pruned_idx = None  # 选中的那个被剪枝token的索引
#         self.neighbor_indices = None   # 它的Top-10邻居索引
#         self.history = []              # 存储各层数据: {'layer': idx, 'pruned_state': tensor, 'neighbor_states': tensor}

#     def select_targets(self, removed_indices, kept_indices, dists, batch_idx=0):
#         """
#         在剪枝层调用一次，随机选一个被剪枝的token并找到其Top-10邻居
#         dists: [R, K] 的距离矩阵
#         """
#         if self.target_pruned_idx is not None:
#             return

#         # 1. 随机选一个 removed token 的相对索引
#         num_removed = removed_indices.shape[1] if len(removed_indices.shape)>1 else len(removed_indices)
#         rel_idx = torch.randint(0, num_removed, (1,)).item()
        
#         # 2. 获取该 token 在原始序列中的绝对索引
#         self.target_pruned_idx = removed_indices[batch_idx, rel_idx].item()

#         # 3. 找到距离该 token 最近的 10 个 kept token 的绝对索引
#         # dists[rel_idx] 对应这个 token 到所有 kept token 的距离
#         _, topk_rel_kept_indices = dists[rel_idx].topk(min(10, dists.shape[1]), largest=False)
#         self.neighbor_indices = kept_indices[batch_idx, topk_rel_kept_indices].cpu().numpy().tolist()

#     def collect(self, layer_idx, hidden_states, batch_idx=0):
#         """
#         每一层(>=K)调用一次，收集这11个token的hidden state
#         注意：这里假设 hidden_states 是全量未剪枝的 (B, N, C)
#         """
#         if self.target_pruned_idx is None:
#             return
        
#         # 提取被剪枝 token 的状态
#         p_state = hidden_states[batch_idx, self.target_pruned_idx, :].detach().cpu().float()
#         # 提取10个邻居的状态
#         n_states = hidden_states[batch_idx, self.neighbor_indices, :].detach().cpu().float()
        
#         self.history.append({
#             'layer': layer_idx,
#             'pruned_state': p_state,      # (C,)
#             'neighbor_states': n_states    # (10, C)
#         })

#     def generate_pca_table(self):
#         """
#         处理收集到的所有数据，运行PCA并返回DataFrame
#         """
#         if not self.history:
#             return None

#         # 1. 汇总所有层的所有状态进行统一 PCA，保证坐标系一致
#         all_vectors = []
#         for h in self.history:
#             all_vectors.append(h['pruned_state'].numpy())
#             all_vectors.extend(h['neighbor_states'].numpy())
        
#         all_vectors = np.array(all_vectors) # (Layers * 11, C)
        
#         # 2. 降维到 2D
#         pca = PCA(n_components=2)
#         reduced_vectors = pca.fit_transform(all_vectors)
        
#         # 3. 重新拆分回各层进行统计
#         results = []
#         pointer = 0
#         for h in self.history:
#             # 提取当前层的 PCA 结果
#             p_v = reduced_vectors[pointer]
#             pointer += 1
#             n_vs = reduced_vectors[pointer : pointer + 10]
#             pointer += 10
            
#             # 计算邻居的均值坐标
#             mean_n_v = n_vs.mean(axis=0)
            
#             results.append({
#                 'Layer': h['layer'],
#                 'pruned_x': p_v[0],
#                 'pruned_y': p_v[1],
#                 'mean_x': mean_n_v[0],
#                 'mean_y': mean_n_v[1]
#             })
        
#         return pd.DataFrame(results)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import uuid # 用于生成唯一文件名

class TokenTrajectoryAnalyzer:
    def __init__(self, save_dir="pca_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.reset()

    def reset(self):
        self.target_batch_id = 0
        self.target_token_idx = None
        self.neighbor_locations = None # 存储 (batch_id, local_token_idx) 的列表
        self.history = []

    def select_targets(self, removed_indices_b0, kept_indices_all, rem_to_kept_idx_global):
        """
        removed_indices_b0: 你的 removed_indexs_per_batch[0]
        kept_indices_all: 你的 keep_indexs_per_batch (B, K_num)
        rem_to_kept_idx_global: 你刚才提到的那个 topk 结果 [R, 10]
        """
        #print("removed_indices_b0",removed_indices_b0, kept_indices_all, rem_to_kept_idx_global)
        if self.target_token_idx is not None: return

        # 1. 随机选一个被删 token 的相对位置 (0 ~ R-1)
        rel_idx = torch.randint(0, len(removed_indices_b0), (1,)).item()
        
        # 2. 锁定目标：Batch 0 中的绝对索引
        self.target_token_idx = removed_indices_b0[rel_idx].item()
        self.target_batch_id = 0

        # 3. 直接拿你算好的 Top-10 全局索引
        target_topk_global = rem_to_kept_idx_global[rel_idx] # [10]
        
        # 4. 换算位置
        B, K_num = kept_indices_all.shape
        self.neighbor_locations = []
        for g_idx in target_topk_global.cpu().numpy():
            b_id = g_idx // K_num
            local_ptr = g_idx % K_num
            actual_token_idx = kept_indices_all[b_id, local_ptr].item()
            self.neighbor_locations.append((b_id, actual_token_idx))

        print("target_token_idx", self.target_token_idx,self.neighbor_locations)

    def collect(self, layer_idx, hidden_states):
        """
        hidden_states: (B, N, C) 每一层的全量状态
        """
        if self.target_token_idx is None: 
            return
        
        # 提取目标 token (固定 Batch 0)
        # 注意：这里我们拿的是 detach 后的副本，不会影响梯度
        p_state = hidden_states[0, self.target_token_idx, :].detach().cpu().float()
        
        # 提取跨 Batch 分布的 10 个邻居
        n_list = []
        for b_id, t_id in self.neighbor_locations:
            n_list.append(hidden_states[b_id, t_id, :].detach().cpu().float())
        
        n_states = torch.stack(n_list)  # (10, C)
        self.history.append({'layer': layer_idx, 'p': p_state, 'n': n_states})

    def finalize_and_save(self):
        if not self.history: 
            return
        
        # 1. 汇总所有层的数据进行统一 PCA
        # 这样做是为了让不同层的坐标在同一个空间内，轨迹才有意义
        all_vecs = []
        for h in self.history:
            all_vecs.append(h['p'].numpy())      # 目标 token
            all_vecs.extend(h['n'].numpy())     # 10个邻居
        
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(np.array(all_vecs))
        
        # 2. 整理数据 (修正了这里的语法错误)
        res = [] 
        ptr = 0
        for h in self.history:
            pv = reduced[ptr]
            ptr += 1
            nv = reduced[ptr : ptr + 10]
            ptr += 10
            
            res.append({
                'Layer': h['layer'], 
                'pruned_x': pv[0], 
                'pruned_y': pv[1], 
                'mean_x': nv.mean(axis=0)[0], 
                'mean_y': nv.mean(axis=0)[1]
            })
        
        df = pd.DataFrame(res)
        
        # 3. 保存文件
        run_id = str(uuid.uuid4())[:8]
        csv_path = os.path.join(self.save_dir, f"trace_{run_id}.csv")
        df.to_csv(csv_path, index=False)
        
        # 4. 绘图
        plt.figure(figsize=(8, 6))
        # 绘制目标 Token 的红色实线轨迹
        plt.plot(df['pruned_x'], df['pruned_y'], 'ro-', label='Target (Batch 0)')
        # 绘制邻居均值的蓝色虚线轨迹
        plt.plot(df['mean_x'], df['mean_y'], 'bo--', label='Neighbors Mean (Global)')
        
        # 标注层数
        for _, r in df.iterrows():
            plt.text(r['pruned_x'], r['pruned_y'], f" L{int(r['Layer'])}", color='red')
            plt.text(r['mean_x'], r['mean_y'], f" L{int(r['Layer'])}", color='blue')
            
        plt.title(f"Feature Trajectory PCA (ID: {run_id})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_path = os.path.join(self.save_dir, f"plot_{run_id}.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f">>> Analysis results saved to: {self.save_dir} (ID: {run_id})")
        
        # 5. 重置状态，准备下一个 Batch 的记录
        self.reset()


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
import torch.nn.functional as F

class EvolutionConsistencyAnalyzer:
    def __init__(self, save_dir="evolution_analysis", target_samples=200):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.target_samples = target_samples
        self.sample_count = 0
        
        # 全局统计池
        self.all_pairwise_l1 = []
        self.all_pairwise_cos = []
        self.all_ngr_l1 = []
        self.all_ngr_cos = []
        
        self.reset_current_step()

    def reset_current_step(self):
        """每张图片处理完后重置临时变量"""
        self.k_states = None
        self.pruned_indices = None
        self.kept_indices = None
        self.neighbor_top10_global = None

    def record_k_layer(self, hidden_states, pruned_idx, kept_idx, neighbor_top10_global):
        """
        在剪枝层调用。
        neighbor_top10_list: rem_to_kept_idx_patch (List of [R, 10])
        """
        self.k_states = hidden_states.detach().cpu().float() # [B, N, C]
        self.pruned_indices = pruned_idx.detach().cpu() # List of [R]
        self.kept_indices = kept_idx.detach().cpu()     # List of [K]
        # 将 Patch 列表堆叠为 Tensor [B, R, 10]
        self.neighbor_top10_global = neighbor_top10_global.detach().cpu()#torch.stack(neighbor_top10_list).detach().cpu()

    def record_final_layer(self, hidden_states):
        """
        在最后一层调用，执行全局 Delta 计算并累加到统计池
        """
        if self.k_states is None: return
        
        final_states = hidden_states.detach().cpu().float()
        B, N, C = final_states.shape
        
        # 1. 计算所有 token 的 Delta (h_final - h_k)
        delta_all = final_states - self.k_states # [B, N, C]
        
        # 2. 构造全局 Kept Delta 池 (跨 Patch 索引的基础)
        # 将所有 Patch 的保留 token 拼在一起
        d_kept_list = [delta_all[b, self.kept_indices[b]] for b in range(B)]
        d_kept_all = torch.cat(d_kept_list, dim=0) # [B*K, C]

        # 3. 统计当前图片的所有数据
        for b in range(B):
            d_pruned = delta_all[b, self.pruned_indices[b]] # [R, C]
            
            # # --- Pairwise (背景分布): 跨 Patch 随机采样 ---
            
            # # 1. 全局 L1 距离 [R, B*K]
            # dist_l1 = torch.cdist(d_pruned, d_kept_all, p=1)
            # # 展平并随机抽取 200 个点
            # flat_l1 = dist_l1.view(-1)
            # # 确保采样数不超过实际总量
            # num_samples = min(200, flat_l1.numel())
            # indices = torch.randperm(flat_l1.numel())[:num_samples]
            # self.all_pairwise_l1.extend(flat_l1[indices].tolist())
            
            # # 2. 全局 Cosine 相似度 [R, B*K]
            # norm_p = F.normalize(d_pruned, p=2, dim=-1)
            # norm_k_all = F.normalize(d_kept_all, p=2, dim=-1)
            # dist_cos = torch.mm(norm_p, norm_k_all.t())
            
            # flat_cos = dist_cos.view(-1)
            # # 采样相同的索引以保持一致性（可选）
            # self.all_pairwise_cos.extend(flat_cos[indices].tolist())

            # --- Pairwise (背景分布): 跨 Patch 随机采样 ---

            # 1. 先随机抽样索引 (核心改动：在计算前抽样)
            num_samples = 200
            # 从 R 和 B*K 中各随机抽 200 个下标组成“随机对”
            idx_p = torch.randint(0, d_pruned.shape[0], (num_samples,))
            idx_k = torch.randint(0, d_kept_all.shape[0], (num_samples,))

            # 2. 只计算这 200 个对的距离 (不再使用 cdist/mm)
            # L1 距离: 对应位置相减
            dist_l1_samples = torch.norm(d_pruned[idx_p] - d_kept_all[idx_k], p=1, dim=-1)
            self.all_pairwise_l1.extend(dist_l1_samples.tolist())

            # Cosine 相似度: 对应位置算余弦
            dist_cos_samples = F.cosine_similarity(d_pruned[idx_p], d_kept_all[idx_k], dim=-1)
            self.all_pairwise_cos.extend(dist_cos_samples.tolist())

            

            # --- NGR (目标分布): 被删 vs 其 Top-10 全局邻居的均值 ---
            for i in range(d_pruned.shape[0]):
                top10_global_idx = self.neighbor_top10_global[b, i]
                d_neighbors_mean = d_kept_all[top10_global_idx].mean(dim=0) # [C]
                
                # L1 差值
                l1_val = torch.norm(d_pruned[i] - d_neighbors_mean, p=1).item()
                # Cos 相似度
                cos_val = F.cosine_similarity(d_pruned[i].unsqueeze(0), d_neighbors_mean.unsqueeze(0)).item()
                
                self.all_ngr_l1.append(l1_val)
                self.all_ngr_cos.append(cos_val)

        self.sample_count += 1
        print(f"Collected sample {self.sample_count}/{self.target_samples}")
        
        # 自动触发保存
        if self.sample_count >= self.target_samples:
            self.generate_results()
        
        self.reset_current_step()

    def generate_results(self):
        """计算均值、中位数，保存表格和图片"""
        run_id = str(uuid.uuid4())[:6]
        summary_stats = []

        for name, pair_data, ngr_data, is_cos in [
            ("L1", self.all_pairwise_l1, self.all_ngr_l1, False),
            ("Cosine", self.all_pairwise_cos, self.all_ngr_cos, True)
        ]:
            pair_arr = np.array(pair_data)
            ngr_arr = np.array(ngr_data)
            
            # 1. 计算分桶数据 (60 bins)
            counts, bin_edges = np.histogram(pair_arr, bins=60, density=True)
            df = pd.DataFrame({
                'bin_left': bin_edges[:-1],
                'bin_right': bin_edges[1:],
                'density': counts
            })
            df.to_csv(os.path.join(self.save_dir, f"{name}_dist_{run_id}.csv"), index=False)

            # 2. 计算统计量
            ngr_median = np.median(ngr_arr)
            ngr_mean = np.mean(ngr_arr)
            summary_stats.append(f"{name} NGR - Mean: {ngr_mean:.4f}, Median: {ngr_median:.4f}")

            # 3. 绘图
            plt.figure(figsize=(8, 5))
            plt.bar(df['bin_left'], df['density'], width=(bin_edges[1]-bin_edges[0]), 
                    color='skyblue', alpha=0.7, label='Pairwise Background')
            plt.axvline(ngr_median, color='orange', linestyle='--', linewidth=2, 
                        label=f'NGR Median: {ngr_median:.4f}')
            
            plt.title(f"Evolution Consistency ({name})")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, f"{name}_plot_{run_id}.png"))
            plt.close()

        # # 1. 打印到屏幕，防止文件没保存成功也能看到
        # print(f"\n" + "*"*50)
        # print(f"RESULT FOR NGR (ID: {run_id})")
        # print(f"L1 - Mean: {ngr_l1_mean:.6f}, Median: {ngr_l1_median:.6f}")
        # print(f"Cos - Mean: {ngr_cos_mean:.6f}, Median: {ngr_cos_median:.6f}")
        # print("*"*50 + "\n")

        # # 2. 写入一个简单的文本文件
        # summary_path = os.path.join(self.save_dir, f"ngr_stats_{run_id}.txt")
        # with open(summary_path, "w") as f:
        #     f.write(f"NGR_L1_Mean: {ngr_l1_mean}\n")
        #     f.write(f"NGR_L1_Median: {ngr_l1_median}\n")
        #     f.write(f"NGR_Cos_Mean: {ngr_cos_mean}\n")
        #     f.write(f"NGR_Cos_Median: {ngr_cos_median}\n")

        # 保存文本摘要
        with open(os.path.join(self.save_dir, f"summary_{run_id}.txt"), "w") as f:
            f.write("\n".join(summary_stats))
        
        print(f"Done! 200 samples analyzed. Files saved in {self.save_dir}")

class InternVisionModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    config_class = InternVisionConfig
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class InternVisionModel_Sparse(InternVisionModel):
    def __init__(self, config: InternVisionConfig):
        super().__init__(config)

        self.encoder = InternVisionEncoder_Sparse(config)