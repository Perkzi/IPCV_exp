# Copyright 2024 The Qwen team and PACT (orailix/PACT, CVPR 2025). SPDX-License-Identifier: Apache-2.0
#
# LLM-stage PACT token pruning + DBDPC clustering for Qwen2-VL, aligned with:
# https://github.com/orailix/PACT/blob/main/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# https://github.com/orailix/PACT/blob/main/transformers/PACT/utils.py
#
# Defaults follow official configs/pact.json; runtime behavior is driven by config.DART_config (same keys as DART).
#
# LLM PACT runs only when DART_config["Sparse"] is True. Recommended keys:
#   - Sparse, pruned_layer, image_token_start_index, image_token_length, reduction_ratio
# Optional overrides:
#   - pact_keep_ratio: stage-1 keep fraction (0–1), highest priority.
#   - pact_reduction_ratio_is_keep_ratio: if True, ``reduction_ratio`` means **retention** (keep), not removal.
#   - pact_use_dbdpc: False → only score pruning, no DBDPC merge (final image token count ≈ stage-1).
#     DBDPC merges similar *kept* representations into cluster means; when enabled, it matters for compression
#     after stage-1. If you disable it, quality/speed tradeoff changes but the model still runs.
#   - pact_cutoff: DBDPC cutoff; smaller → less aggressive merge → more tokens kept after merge.
#   - pact_compensate_dbdpc_final_reduction: if True, treat ``reduction_ratio`` as **target after DBDPC**
#     (removed fraction if ``pact_reduction_ratio_is_keep_ratio`` is False; see docstring). Stage-1 keep is
#     ``min(1, T_final_keep * pact_dbdpc_compensate_mult)`` so you prune less before merge; tune **mult** (e.g. 3–8)
#     until measured post-DBDPC keep ≈ T_final_keep.
#   - pact_final_image_keep_ratio: if set and > 0, after DBDPC keep at most ``max(1, round(image_len * ratio))``
#     image-side rows (top-k by L2 norm of merged states). Overrides auto final trim below.
#   - pact_auto_final_keep_from_reduction: default **True** when ``pact_final_image_keep_ratio`` is unset. If True,
#     after DBDPC apply the same cap using nominal keep = ``pact_keep_ratio`` or ``1 - reduction_ratio``
#     (or ``reduction_ratio`` when ``pact_reduction_ratio_is_keep_ratio``), so final image count tracks your
#     ``reduction_ratio`` / eval script. Set **False** to skip this post-trim (legacy: keep all DBDPC centers).
#   - pact_enforce_nominal_stage1_min_keep: default **True**. Stage-1 top-k keep is at least
#     ``round(num_image_scores * nominal_keep)`` where ``nominal_keep`` comes from ``pact_keep_ratio`` or
#     ``reduction_ratio`` (ignores ``pact_compensate_dbdpc_final_reduction`` inflation). Stops ``int(N*p)==0``
#     from collapsing to a single image token when compensation makes ``pruning_tokeep_percentage_value`` tiny.
#     Set **False** for legacy behavior.
#
# Default (no compensation): ``reduction_ratio`` = fraction of **image** tokens removed at **stage 1**;
# keep = 1 - reduction_ratio. DBDPC may change counts; optional post-trim aligns final keep with the same nominal ratio.

from __future__ import annotations

import math
import time
from copy import copy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from .modeling_qwen2_vl_base import (
    DART_ViT,
    Qwen2VLForConditionalGeneration as Qwen2VLFCBase,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

logger = logging.get_logger(__name__)


def _build_pact_config() -> SimpleNamespace:
    """Mirrors orailix/PACT load_config defaults where pact.json is silent."""
    cfg = SimpleNamespace(
        visual_token_reduction=True,
        layer_for_reduction=4,
        progessive_reduction=False,
        use_DBDPC=False,
        cutoff=0.21,
        vector_to_use_in_distance_clustering="current_k_cosine",
        take_mean=True,
        include_pruned_in_mean=True,
        do_not_consider_non_image_tokens_as_pruned=True,
        coef_pruned=1.5,
        avoid_numerical_instability_DBDPC=True,
        withdraw_visual_tokens=False,
        VTW_equivalant_layer_for_reduction=-1,
        equivalent_reduc_percentage_vtw=0.0,
        use_tome=False,
        perc_tokeep_tome_total=1.0,
        tome_equivalant_layer_for_reduction=4,
        use_kmeans=False,
        perc_tokeep_kmeans=1.0,
        use_dpc=False,
        percentage_to_keep_dpc=1.0,
        use_agglomerative=False,
        percentage_to_keep_agglomerative=1.0,
        linkage="single",
        use_dbscan=False,
        eps_dbscan=0.1,
        noise_as_clusters_dbscan=False,
        token_pruning=True,
        use_all_non_text_pruning=True,
        prune_with_norm=False,
        use_cosine_in_token_pruning=False,
        use_attention_in_token_pruning=False,
        use_mask_in_use_attention_in_token_pruning=False,
        use_IQR_in_token_pruning=False,
        alpha_IQR=0.5,
        pruning_filter_wth_percentage=True,
        pruning_tokeep_percentage_value=0.55,
        multiply_by_norm=True,
        norm_to_use=2,
        avoid_numerical_instability_prune=True,
        no_proportional_attention=True,
        change_position_ids=False,
        get_mean_position_id=False,
        synchro=False,
        need_kq=True,
        do_not_upcast_to_full_precision_for_pruning=False,
        keep_casual=True,
        get_performance_metrics=False,
        get_reduction_ratio=True,
        use_custom_merging=False,
        use_custom_pruning=False,
        log_output_path="agg_pact_logs",
    )
    return cfg


def build_pact_namespace(dart_cfg: Optional[Dict[str, Any]]) -> SimpleNamespace:
    """
    Merge official PACT defaults with ``config.DART_config`` (same dict as ``class DART``).

    Stage-1 keep ratio (``pruning_tokeep_percentage_value``):

    1. ``pact_keep_ratio`` if set (always stage-1 only; no auto compensation).
    2. Else if ``reduction_ratio`` set:
       - If ``pact_compensate_dbdpc_final_reduction`` is True and ``use_DBDPC`` is True (after overrides):
         ``T_final`` = target **fraction of image tokens still present after DBDPC**
         (``reduction_ratio`` = removed fraction → ``T_final = 1 - rr`` when
         ``pact_reduction_ratio_is_keep_ratio`` is False; if True, ``T_final = rr``).
         Stage-1 keep = ``min(1, T_final * pact_dbdpc_compensate_mult)``.
       - Else: same as before — ``clamp(rr)`` or ``clamp(1 - rr)`` for stage-1 only.
    3. Else default from ``_build_pact_config()`` (0.55).
    """
    base = copy(vars(_build_pact_config()))
    if dart_cfg is None:
        base["visual_token_reduction"] = False
        return SimpleNamespace(**base)

    if dart_cfg.get("Sparse") is True:
        base["visual_token_reduction"] = True
    else:
        base["visual_token_reduction"] = False

    if "pruned_layer" in dart_cfg:
        base["layer_for_reduction"] = int(dart_cfg["pruned_layer"])

    if dart_cfg.get("pact_use_dbdpc") is not None:
        base["use_DBDPC"] = bool(dart_cfg["pact_use_dbdpc"])

    if dart_cfg.get("pact_keep_ratio") is not None:
        base["pruning_tokeep_percentage_value"] = float(dart_cfg["pact_keep_ratio"])
    elif "reduction_ratio" in dart_cfg:
        rr = float(dart_cfg["reduction_ratio"])
        keep_is_rr = dart_cfg.get("pact_reduction_ratio_is_keep_ratio") is True
        t_final = max(0.0, min(1.0, rr if keep_is_rr else (1.0 - rr)))

        compensate = (
            dart_cfg.get("pact_compensate_dbdpc_final_reduction") is True and base.get("use_DBDPC", True)
        )
        if compensate:
            mult = float(dart_cfg.get("pact_dbdpc_compensate_mult", 3.0))
            mult = max(mult, 1e-6)
            base["pruning_tokeep_percentage_value"] = max(0.0, min(1.0, t_final * mult))
        else:
            base["pruning_tokeep_percentage_value"] = t_final

    if dart_cfg.get("pact_cutoff") is not None:
        base["cutoff"] = float(dart_cfg["pact_cutoff"])

    if dart_cfg.get("pact_multiply_by_norm") is not None:
        base["multiply_by_norm"] = bool(dart_cfg["pact_multiply_by_norm"])

    if dart_cfg.get("pact_include_pruned_in_mean") is not None:
        base["include_pruned_in_mean"] = bool(dart_cfg["pact_include_pruned_in_mean"])

    if dart_cfg.get("pact_coef_pruned") is not None:
        base["coef_pruned"] = float(dart_cfg["pact_coef_pruned"])

    return SimpleNamespace(**base)


def pact_nominal_image_keep_fraction(dart_cfg: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    User-facing image keep fraction from ``pact_keep_ratio`` or ``reduction_ratio`` only.

    Ignores ``pact_compensate_dbdpc_final_reduction`` (that only inflates stage-1 ``pruning_tokeep_percentage_value``).
    """
    if dart_cfg is None:
        return None
    if dart_cfg.get("pact_keep_ratio") is not None:
        return max(0.0, min(1.0, float(dart_cfg["pact_keep_ratio"])))
    if "reduction_ratio" in dart_cfg:
        rr = float(dart_cfg["reduction_ratio"])
        keep_is_rr = dart_cfg.get("pact_reduction_ratio_is_keep_ratio") is True
        return max(0.0, min(1.0, rr if keep_is_rr else (1.0 - rr)))
    return None


def pact_stage1_min_keep_count(num_elements: int, dart_cfg: Optional[Dict[str, Any]]) -> int:
    """Minimum stage-1 image tokens to keep when nominal keep is known (batch 1, percentage pruning)."""
    if dart_cfg is None or num_elements <= 0:
        return 1
    if dart_cfg.get("pact_enforce_nominal_stage1_min_keep") is False:
        return 1
    f = pact_nominal_image_keep_fraction(dart_cfg)
    if f is None or f <= 0.0:
        return 1
    return max(1, min(num_elements, int(round(float(num_elements) * float(f)))))


def pact_resolve_final_image_keep_ratio(dart_cfg: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Target fraction of *original* ``image_token_length`` for the post-DBDPC trim (cap by top-k on merged rows).

    Precedence:
    1. ``pact_final_image_keep_ratio`` if set and > 0.
    2. Else if ``pact_auto_final_keep_from_reduction`` is False → no trim (``None``).
    3. Else same as ``pact_nominal_image_keep_fraction`` (``pact_keep_ratio`` or ``reduction_ratio``).

    Does **not** apply the DBDPC compensation multiplier: final cap matches the user-facing ratio, not stage-1 inflation.
    """
    if dart_cfg is None:
        return None
    explicit = dart_cfg.get("pact_final_image_keep_ratio")
    if explicit is not None:
        v = float(explicit)
        return v if v > 0.0 else None
    if dart_cfg.get("pact_auto_final_keep_from_reduction") is False:
        return None
    return pact_nominal_image_keep_fraction(dart_cfg)


# Backwards-compatible read-only defaults (prefer build_pact_namespace + DART_config).
PACT_CONFIG = _build_pact_config()


# -------------------------- PACT/utils.py (subset): distances, DBDPC, merge --------------------------

def normal_compute_pairwise_distances(X: torch.Tensor, l_2: bool = False) -> torch.Tensor:
    X = F.normalize(X, p=2.0, dim=-1)
    dot_product = torch.mm(X, X.t())
    cosine_distance = 1 - dot_product
    return cosine_distance


class DBDPC:
    def __init__(self, dc: int = 2):
        self.dc = dc

    def get_clusters(self):
        if not hasattr(self, "labels_") or self.labels_ is None:
            raise ValueError("Labels have not been initialized. Run the clustering algorithm first.")
        clusters = {label: np.where(self.labels_ == label)[0] for label in np.unique(self.labels_) if label != -1}
        return clusters

    def fit_variant(self, X, cutoff, pact_config, pruned_keys=None, print_: bool = False):
        device = X.device
        N = X.shape[0]
        cluster_centers: List[int] = []
        unassigned_mask = torch.ones(N, dtype=torch.bool, device=device)

        print_ = print_ and pact_config.synchro

        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            start = time.time()

        dist = torch.clamp(normal_compute_pairwise_distances(X), min=0)
        dist.fill_diagonal_(0)

        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            print(f"Distance matrix computation takes {time.time() - start} seconds")
            print(f"dist max is {dist.max().item()}")

        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            start = time.time()

        rho = torch.exp(-((dist / self.dc) ** 2)).sum(dim=1)

        if pact_config.avoid_numerical_instability_DBDPC:
            sorted_indices = torch.argsort(rho)
            ranks = torch.empty_like(sorted_indices).to(rho.device).to(rho.dtype)
            ranks[sorted_indices] = torch.arange(len(rho), device=rho.device).to(rho.dtype)
            rho = ranks

        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            print(f"Rho calculations take {time.time() - start} seconds")

        iteration = 0
        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            start = time.time()

        while True:
            iteration += 1
            if print_:
                print(f"\nIteration {iteration}: Starting with {unassigned_mask.sum().item()} unassigned points.")

            unassigned_indices = torch.where(unassigned_mask)[0]
            num_unassigned = unassigned_indices.size(0)
            if num_unassigned == 0:
                break

            dist_unassigned = dist[unassigned_indices][:, unassigned_indices]
            rho_unassigned = rho[unassigned_indices]

            rho_expand = rho_unassigned.unsqueeze(1).expand(num_unassigned, num_unassigned)

            rho_compare = ~torch.gt(rho_expand, rho_expand.t())
            rho_compare.fill_diagonal_(False)

            inf_mask = torch.full_like(dist_unassigned, float("inf"), dtype=dist_unassigned.dtype)
            conditioned_spatial_dist_matrix = torch.where(rho_compare, dist_unassigned, inf_mask)

            delta_unassigned, _ = torch.min(conditioned_spatial_dist_matrix, dim=1)

            max_rho = rho_unassigned.max()
            max_rho_mask = rho_unassigned == max_rho
            delta_unassigned[max_rho_mask] = float("inf")

            new_centers_mask = delta_unassigned > cutoff
            new_centers_indices = unassigned_indices[new_centers_mask]

            num_new_centers = new_centers_indices.numel()
            if num_new_centers == 0:
                if print_:
                    print("No new cluster centers found.")
                break

            cluster_centers.extend(new_centers_indices.tolist())

            if print_:
                print(f"Identified {num_new_centers} new cluster centers.")

            dist_to_new_centers = dist[unassigned_indices][:, new_centers_indices]
            within_cutoff = (dist_to_new_centers <= cutoff).any(dim=1)

            points_within_cutoff = unassigned_indices[within_cutoff]
            unassigned_mask[points_within_cutoff] = False
            unassigned_mask[new_centers_indices] = False

            if num_new_centers < 10:
                if print_:
                    print("Fewer than 10 new centers identified, reverting to previous method.")

                delta = delta_unassigned.clone()
                delta_sorted_indices = unassigned_indices[torch.argsort(-delta_unassigned)]
                delta_sorted_values = delta_unassigned[torch.argsort(-delta_unassigned)]

                if pact_config.synchro and print_:
                    torch.cuda.synchronize()
                    start = time.time()

                first_index = torch.searchsorted(-delta_sorted_values, -cutoff)

                cluster_centers.extend(delta_sorted_indices[:first_index].tolist())

                delta_sorted_indices = delta_sorted_indices[first_index:]

                rho_unassigned_sorted = rho[delta_sorted_indices]
                delta_sorted_indices = delta_sorted_indices[torch.argsort(-rho_unassigned_sorted)]

                dist_to_centers = dist[delta_sorted_indices][:, cluster_centers]

                not_within_cutoff_all = (dist_to_centers > cutoff).all(dim=1)

                delta_sorted_indices = delta_sorted_indices[not_within_cutoff_all]

                dist_mapped = dist[delta_sorted_indices][:, delta_sorted_indices]

                within_cutoff = dist_mapped <= cutoff
                within_cutoff.fill_diagonal_(False)

                assigned_mask = torch.zeros(dist_mapped.shape[0], dtype=torch.bool, device=dist.device)

                within_cutoff_any = within_cutoff.any(dim=0)

                for index, i in enumerate(delta_sorted_indices):
                    if not assigned_mask[index]:
                        cluster_centers.append(i.item())
                        if within_cutoff_any[index]:
                            assigned_mask[index + 1 :] |= within_cutoff[index, index + 1 :]

                break

            if pact_config.synchro and print_:
                torch.cuda.synchronize()
                if print_:
                    print(f"New center recursive identification calculations take {time.time() - start} seconds")

        # Degenerate DBDPC (0 or 1 distinct center) makes every kept token share one label after nearest-center
        # assignment → merge_clusters keeps a single row → ~1 image token left. Upstream utils.py can hit the
        # same when ``min_distances <= cutoff * coef_pruned`` leaves most labels at -1. Fallback: identity centers.
        if len(cluster_centers) == 0:
            cluster_centers_t = torch.arange(N, device=device, dtype=torch.long)
        else:
            cluster_centers_t = torch.tensor(cluster_centers, device=device, dtype=torch.long)
            cluster_centers_t = torch.unique(cluster_centers_t)
        if cluster_centers_t.numel() == 1 and N > 1:
            logger.warning_once(
                "DBDPC produced a single cluster center; using per-token centers for this forward (no merge)."
            )
            cluster_centers_t = torch.arange(N, device=device, dtype=torch.long)

        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            start = time.time()

        if pruned_keys is None:
            dist_to_centers = dist[:, cluster_centers_t]
            nearest_center = torch.argmin(dist_to_centers, dim=1)
            labels = cluster_centers_t[nearest_center]
        else:
            X_normalized = F.normalize(X, p=2.0, dim=1)
            pruned_keys_normalized = F.normalize(pruned_keys, p=2.0, dim=1)

            num_initial = X_normalized.size(0)
            num_pruned = pruned_keys_normalized.size(0)
            num_tokens = num_initial + num_pruned

            cluster_center_vectors = X_normalized[cluster_centers_t]

            dist_to_centers_xuntouched = dist[:, cluster_centers_t]
            # Kept tokens: always assign to nearest center (same as pruned_keys is None). Using a global
            # ``min_distances <= cutoff`` mask for *all* rows left most kept rows at label -1, so merge only
            # saw singleton “clusters” at center indices → second_mask collapsed to ~1 True.
            nearest_kept = torch.argmin(dist_to_centers_xuntouched, dim=1)
            labels = torch.full((num_tokens,), -1, dtype=torch.long, device=device)
            labels[:num_initial] = cluster_centers_t[nearest_kept]

            similarity_pruned = torch.matmul(pruned_keys_normalized, cluster_center_vectors.transpose(0, 1))
            dist_pruned = 1.0 - similarity_pruned
            min_pruned, nearest_pruned = torch.min(dist_pruned, dim=1)
            cutoff_assign = cutoff * pact_config.coef_pruned
            mask_pruned = min_pruned <= cutoff_assign
            if num_pruned > 0:
                tail = labels[num_initial:]
                tail[mask_pruned] = cluster_centers_t[nearest_pruned[mask_pruned]]
            labels[cluster_centers_t] = cluster_centers_t

        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            if print_:
                print(f"Point assignment to nearest cluster centers takes {time.time() - start} seconds")

        self.labels_ = labels.cpu().numpy()


def compute_fastest_cluster_means_with_arbitrary_ids(
    tensor: torch.Tensor,
    clusters: dict,
    weights_list=None,
    synchro: bool = False,
    print_: bool = False,
):
    if synchro and print_:
        torch.cuda.synchronize()
        start = time.time()

    cluster_id_list = list(clusters.keys())
    id_to_index = {cluster_id: i for i, cluster_id in enumerate(cluster_id_list)}

    num_clusters = len(cluster_id_list)
    feature_dim = tensor.size(1)

    cluster_sums = torch.zeros((num_clusters, feature_dim), device=tensor.device, dtype=tensor.dtype)

    all_indices_list = []
    for indices in clusters.values():
        all_indices_list.extend(indices.tolist() if hasattr(indices, "tolist") else list(indices))

    all_indices = torch.tensor(all_indices_list, dtype=torch.long, device=tensor.device)

    cluster_indices_list = []
    for cluster_id, indices in clusters.items():
        cluster_indices_list.extend([id_to_index[cluster_id]] * len(indices))

    cluster_indices = torch.tensor(cluster_indices_list, dtype=torch.long, device=tensor.device)

    if weights_list is None:
        cluster_sizes = torch.zeros(num_clusters, device=tensor.device, dtype=tensor.dtype)
        cluster_sizes.index_add_(0, cluster_indices, torch.ones_like(cluster_indices, dtype=tensor.dtype))
    else:
        if isinstance(weights_list, list):
            weights_list = torch.tensor(weights_list, dtype=tensor.dtype, device=tensor.device)
        cluster_sizes = weights_list

    if synchro and print_:
        torch.cuda.synchronize()
        print(f"Preparation for merging took {time.time() - start}")

    cluster_sums.index_add_(0, cluster_indices, tensor[all_indices])

    cluster_means = cluster_sums / cluster_sizes.unsqueeze(1)

    return cluster_means


def merge_clusters(tensor, clusters, pact_config, position_ids=None, pruned_hiddens=None, print_: bool = False):
    if pact_config.synchro and print_:
        torch.cuda.synchronize()
        start = time.time()

    cluster_indices = [cluster_id for cluster_id in clusters.keys()]
    weights_list = [len(indices) for _, indices in clusters.items()]

    if pruned_hiddens is not None:
        tensor_with_pruned = torch.cat([tensor, pruned_hiddens], dim=0)
        if pact_config.take_mean:
            cluster_means = compute_fastest_cluster_means_with_arbitrary_ids(
                tensor_with_pruned, clusters, weights_list, synchro=pact_config.synchro
            )
        else:
            cluster_means = torch.stack([tensor_with_pruned[cluster_id] for cluster_id in clusters.keys()])
    else:
        if pact_config.take_mean:
            cluster_means = compute_fastest_cluster_means_with_arbitrary_ids(
                tensor, clusters, weights_list, synchro=pact_config.synchro
            )
        else:
            cluster_means = torch.stack([tensor[cluster_id] for cluster_id in clusters.keys()])

    if pact_config.synchro and print_:
        torch.cuda.synchronize()
        print(f"Merging calculation took {time.time() - start}")

    output_tensor = torch.zeros_like(tensor, dtype=tensor.dtype, device=tensor.device)
    mask = torch.zeros((tensor.size(0), 1), dtype=torch.bool, device=tensor.device)

    weights = torch.zeros_like(tensor[:, 0], dtype=torch.float32, device=tensor.device)
    weights_centers = torch.tensor(weights_list, dtype=torch.float32, device=tensor.device)
    weights[cluster_indices] = weights_centers

    output_tensor[cluster_indices] = cluster_means
    mask[cluster_indices, 0] = True

    mask = mask.to(torch.bool)

    if pact_config.get_mean_position_id:
        position_ids_output = torch.zeros_like(position_ids)
        for cluster_id, indices in clusters.items():
            values = position_ids[indices]
            mean_val = torch.round(values.float().mean()).long()
            position_ids_output[indices] = mean_val
        return output_tensor, mask, weights, position_ids_output
    return output_tensor, mask, weights, None


def token_reduction(
    image_feature,
    image_feature_for_clustering,
    cutoff,
    reduction,
    pact_config,
    position_ids=None,
    pruned_hiddens=None,
    pruned_keys=None,
):
    dbdpc_variant = DBDPC(dc=2)
    dbdpc_variant.fit_variant(image_feature_for_clustering, cutoff, pact_config=pact_config, pruned_keys=pruned_keys)
    clusters_variant = dbdpc_variant.get_clusters()

    merged, mask_image, weights, position_ids_output = merge_clusters(
        image_feature,
        clusters_variant,
        pact_config,
        position_ids=position_ids,
        pruned_hiddens=pruned_hiddens,
    )
    return merged, mask_image, weights, position_ids_output


def pact_trim_merged_tokens(
    merged: torch.Tensor,
    second_mask: torch.Tensor,
    target_count: int,
    position_ids_after_reduction: Optional[torch.Tensor] = None,
):
    """
    If DBDPC leaves more than ``target_count`` cluster centers (True in ``second_mask``), keep the
    top ``target_count`` by L2 norm of ``merged`` rows. Does not change DBDPC itself; only post-filters.
    """
    sm = second_mask.reshape(-1).bool()
    idx = torch.where(sm)[0]
    if idx.numel() <= target_count:
        return merged, second_mask, position_ids_after_reduction

    topk = min(int(target_count), int(idx.numel()))
    sub = merged[idx].to(torch.float32)
    scores = torch.norm(sub, p=2, dim=-1)
    keep_within = idx[scores.topk(topk).indices]

    new_merged = torch.zeros_like(merged)
    new_merged[keep_within] = merged[keep_within]

    new_sm_flat = torch.zeros_like(sm)
    new_sm_flat[keep_within] = True
    new_mask = new_sm_flat.view_as(second_mask)

    new_pos = position_ids_after_reduction
    if position_ids_after_reduction is not None:
        new_pos = torch.zeros_like(position_ids_after_reduction)
        new_pos[keep_within] = position_ids_after_reduction[keep_within]

    return new_merged, new_mask, new_pos


def pact_compute_qk_for_scores(
    hidden_states: torch.Tensor,
    decoder_layer,
    position_ids: torch.Tensor,
    cache_position: Optional[torch.Tensor],
    past_key_value,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match PACT get_key_query_value / Qwen2VLAttention projections + RoPE."""
    attn = decoder_layer.self_attn
    bsz, q_len, _ = hidden_states.size()

    hidden_states_normalized = decoder_layer.input_layernorm(hidden_states)

    current_k = attn.k_proj(hidden_states_normalized)
    current_q = attn.q_proj(hidden_states_normalized)
    value_states = attn.v_proj(hidden_states_normalized)

    current_k_cosine = current_k.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    current_q_cosine = current_q.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)

    kv_seq_len = current_k_cosine.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, attn.layer_idx)

    rotary_seq_len = (
        max(kv_seq_len, position_ids[:, -1].max().item() + 1) if position_ids is not None else kv_seq_len
    )

    cos, sin = attn.rotary_emb(value_states, seq_len=rotary_seq_len)

    current_k_cosine, current_q_cosine = apply_multimodal_rotary_pos_emb(
        current_k_cosine, current_q_cosine, cos, sin, position_ids, attn.rope_scaling["mrope_section"]
    )

    current_k_cosine = current_k_cosine.transpose(1, 2).flatten(2, 3)
    current_q_cosine = current_q_cosine.transpose(1, 2).flatten(2, 3)

    return current_k, current_q, current_k_cosine, current_q_cosine


class PACT(Qwen2VLModel):
    """
    Qwen2-VL language tower with PACT (pruning + DBDPC) when ``config.DART_config['Sparse']`` is True.

    Uses ``pruned_layer``, ``image_token_start_index``, ``image_token_length``, ``reduction_ratio`` (same as ``DART``).
    Stage-1 keep ratio = ``1 - reduction_ratio`` unless ``pact_keep_ratio`` is set.
    After DBDPC, by default ``pact_resolve_final_image_keep_ratio`` caps image-side tokens to the same nominal keep
    fraction of *original* ``image_token_length`` (see module docstring).

    Batch size 1; reduction on prefill only (no KV cache).
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        dart_cfg = getattr(self.config, "DART_config", None)
        pc = build_pact_namespace(dart_cfg)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, _seq_length_in = inputs_embeds.shape[:2]
        assert batch_size == 1, "PACT path expects batch_size == 1"

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        reduction_stats = [0, 0] if pc.get_reduction_ratio else None

        real_position_ids = position_ids
        seq_len = hidden_states.shape[1]

        # VTW-equivalent layer remap (same as upstream PACT)
        layer_for_reduction = pc.layer_for_reduction
        if pc.VTW_equivalant_layer_for_reduction != -1:
            reduction_perc_total = 1 - (1 - pc.equivalent_reduc_percentage_vtw) * (
                (len(self.layers) - pc.VTW_equivalant_layer_for_reduction) / len(self.layers)
            )
            layer_for_reduction = round(reduction_perc_total * len(self.layers))

        prefill = past_key_values is None or past_key_values.get_seq_length() == 0

        if dart_cfg is not None:
            image_start = int(dart_cfg["image_token_start_index"])
            image_len = int(dart_cfg["image_token_length"])
        else:
            image_start, image_len = 0, 0

        for index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            apply_reduction = (
                pc.visual_token_reduction
                and prefill
                and index == layer_for_reduction
                and image_len > 0
                and pc.need_kq
            )

            if apply_reduction:
                device = hidden_states.device
                is_image = torch.zeros(seq_len, dtype=torch.bool, device=device)
                if image_start + image_len <= seq_len:
                    is_image[image_start : image_start + image_len] = True
                else:
                    logger.warning_once(
                        "image_token_start_index/length outside sequence; skipping PACT for this forward."
                    )
                    apply_reduction = False

            if apply_reduction:
                is_image_indices = torch.where(is_image)[0]
                if is_image_indices.numel() == 0:
                    apply_reduction = False
                else:
                    last_true_idx = is_image_indices.max()
                    is_not_text = torch.zeros(seq_len, dtype=torch.bool, device=device)
                    is_not_text[: last_true_idx + 1] = True

                if apply_reduction and pc.get_performance_metrics:
                    torch.cuda.synchronize()
                    start_algo = time.time()

                if apply_reduction:
                    current_k, current_q, current_k_cosine, current_q_cosine = pact_compute_qk_for_scores(
                        hidden_states, decoder_layer, position_ids, cache_position, past_key_values
                    )

                    if pc.vector_to_use_in_distance_clustering == "current_k_cosine":
                        vector_to_use = current_k_cosine
                    elif pc.vector_to_use_in_distance_clustering == "current_q_cosine":
                        vector_to_use = current_q_cosine
                    else:
                        vector_to_use = current_k_cosine

                    attn = decoder_layer.self_attn
                    head_dim = attn.head_dim

                    stage1_num_to_keep_for_log: Optional[int] = None

                    # ---------- Token pruning scores (aligned with orailix/PACT modeling_qwen2_vl) ----------
                    if pc.token_pruning:
                        if pc.use_cosine_in_token_pruning:
                            current_k_sel = current_k_cosine[:, is_not_text]
                            current_q_sel = current_q_cosine[:, is_not_text]
                        else:
                            if pc.use_all_non_text_pruning:
                                current_k_sel = current_k[:, is_not_text]
                                current_q_sel = current_q[:, is_not_text]
                            else:
                                current_k_sel = current_k[:, is_image]
                                current_q_sel = current_q[:, is_not_text]

                        current_k_image = current_k_sel.view(
                            batch_size, -1, attn.num_key_value_heads, attn.head_dim
                        ).transpose(1, 2)
                        current_k_image = repeat_kv(current_k_image, attn.num_key_value_groups)
                        current_q_image = current_q_sel.view(
                            batch_size, -1, attn.num_heads, attn.head_dim
                        ).transpose(1, 2)

                        global_q = torch.mean(current_q_image, dim=2, keepdim=True)
                        if pc.do_not_upcast_to_full_precision_for_pruning:
                            scores = torch.matmul(
                                global_q / math.sqrt(head_dim), current_k_image.transpose(-2, -1)
                            )
                        else:
                            scores = torch.matmul(
                                global_q.to(torch.float32) / math.sqrt(head_dim),
                                current_k_image.to(torch.float32).transpose(-2, -1),
                            )

                        scores = torch.softmax(scores, dim=-1, dtype=torch.float32).mean(1)
                        scores = torch.nan_to_num(scores, nan=0.0)

                        if pc.use_attention_in_token_pruning:
                            scores = scores.mean(-2)
                        else:
                            scores = scores.squeeze(-2)

                        if pc.use_all_non_text_pruning:
                            scores = scores[:, is_image[is_not_text]]

                        if pc.multiply_by_norm:
                            norm = torch.norm(
                                hidden_states[:, is_image].to(torch.float32),
                                dim=-1,
                                p=pc.norm_to_use,
                            ).squeeze(0)
                            scores = scores.to(torch.float32) * norm

                        if pc.avoid_numerical_instability_prune and pc.pruning_filter_wth_percentage:
                            scores = scores.squeeze(0)
                            sorted_indices = torch.argsort(scores)
                            ranks = torch.empty_like(sorted_indices).to(scores.device).to(scores.dtype)
                            ranks[sorted_indices] = torch.arange(len(scores), device=scores.device).to(scores.dtype)
                            scores = ranks

                        if pc.pruning_filter_wth_percentage:
                            scores = scores.squeeze(0)
                            num_elements = scores.numel()
                            pct = float(pc.pruning_tokeep_percentage_value)
                            raw = float(num_elements) * max(0.0, min(1.0, pct))
                            num_to_keep = min(
                                num_elements,
                                max(1, int(math.ceil(raw - 1e-9))),
                            )
                            num_to_keep = max(
                                num_to_keep,
                                pact_stage1_min_keep_count(num_elements, dart_cfg),
                            )
                            stage1_num_to_keep_for_log = int(num_to_keep)
                            sorted_scores, _ = torch.sort(scores, descending=True)
                            thresh_scores = sorted_scores[num_to_keep - 1]
                            first_mask = scores >= thresh_scores
                        elif pc.use_IQR_in_token_pruning:
                            scores_flat = scores.flatten()
                            q1 = torch.quantile(scores_flat, 0.25)
                            q3 = torch.quantile(scores_flat, 0.75)
                            thresh_scores = q1 + pc.alpha_IQR * (q3 - q1)
                            first_mask = scores >= thresh_scores
                        else:
                            first_mask = torch.ones_like(scores, dtype=torch.bool)

                        first_mask = first_mask.squeeze().bool()
                    elif pc.prune_with_norm:
                        norm = torch.norm(hidden_states[:, is_image].to(torch.float32), dim=-1, p=2).squeeze(0)
                        scores = norm
                        num_elements = scores.numel()
                        pct = float(pc.pruning_tokeep_percentage_value)
                        raw = float(num_elements) * max(0.0, min(1.0, pct))
                        num_to_keep = min(
                            num_elements,
                            max(1, int(math.ceil(raw - 1e-9))),
                        )
                        num_to_keep = max(
                            num_to_keep,
                            pact_stage1_min_keep_count(num_elements, dart_cfg),
                        )
                        stage1_num_to_keep_for_log = int(num_to_keep)
                        sorted_scores, _ = torch.sort(scores, descending=True)
                        thresh_scores = sorted_scores[num_to_keep - 1]
                        first_mask = (scores >= thresh_scores).squeeze().bool()
                    elif pc.withdraw_visual_tokens:
                        first_mask = torch.zeros(is_image.sum().item(), dtype=torch.bool, device=device)
                    else:
                        first_mask = torch.ones(is_image.sum().item(), dtype=torch.bool, device=device)

                    first_mask_global = is_image.clone()
                    first_mask_global[is_image] = first_mask.clone()

                    if reduction_stats is not None:
                        reduction_stats[0] += int(first_mask.sum().item())

                    reduction_list = reduction_stats if reduction_stats is not None else [0, 0]

                    _den = int(first_mask.numel())
                    _kept = int(first_mask.sum().item())
                    _pct = (100.0 * _kept / _den) if _den > 0 else 0.0
                    # print(
                    #     "[PACT] pre-DBDPC stage1 image:",
                    #     f"kept {_kept}/{_den} ({_pct:.2f}%)",
                    #     f"pruning_tokeep_percentage_value={float(pc.pruning_tokeep_percentage_value):.6g}",
                    #     f"num_to_keep_target={stage1_num_to_keep_for_log}",
                    #     f"nominal_keep_frac={pact_nominal_image_keep_fraction(dart_cfg)}",
                    #     f"enforced_min_keep={pact_stage1_min_keep_count(_den, dart_cfg)}",
                    #     f"config_image_token_length={image_len}",
                    # )

                    # ---------- DBDPC merge ----------
                    if pc.use_DBDPC and index == layer_for_reduction:
                        real_position_ids_after_mask_image = real_position_ids.permute(2, 1, 0)[first_mask_global]

                        vec_kept = vector_to_use[:, first_mask_global].squeeze(0)
                        hid_kept = hidden_states[:, first_mask_global].squeeze(0)

                        if not pc.include_pruned_in_mean:
                            merged, second_mask, weights_tok, position_ids_after_reduction = token_reduction(
                                hid_kept,
                                vec_kept,
                                pc.cutoff,
                                reduction_list,
                                pc,
                                position_ids=real_position_ids_after_mask_image,
                                pruned_hiddens=None,
                                pruned_keys=None,
                            )
                        elif pc.do_not_consider_non_image_tokens_as_pruned:
                            pruned_hiddens = hidden_states[:, is_image][:, ~first_mask].squeeze(0)
                            pruned_keys = vector_to_use[:, is_image][:, ~first_mask].squeeze(0)
                            merged, second_mask, weights_tok, position_ids_after_reduction = token_reduction(
                                hid_kept,
                                vec_kept,
                                pc.cutoff,
                                reduction_list,
                                pc,
                                position_ids=real_position_ids_after_mask_image,
                                pruned_hiddens=pruned_hiddens,
                                pruned_keys=pruned_keys,
                            )
                        else:
                            pruned_hiddens = hidden_states[:, ~first_mask_global].squeeze(0)
                            pruned_keys = vector_to_use[:, ~first_mask_global].squeeze(0)
                            merged, second_mask, weights_tok, position_ids_after_reduction = token_reduction(
                                hid_kept,
                                vec_kept,
                                pc.cutoff,
                                reduction_list,
                                pc,
                                position_ids=real_position_ids_after_mask_image,
                                pruned_hiddens=pruned_hiddens,
                                pruned_keys=pruned_keys,
                            )

                        trim_r = pact_resolve_final_image_keep_ratio(dart_cfg)
                        if trim_r is not None and float(trim_r) > 0.0:
                            target_n = max(1, int(round(float(image_len) * float(trim_r))))
                            merged, second_mask, position_ids_after_reduction = pact_trim_merged_tokens(
                                merged,
                                second_mask,
                                target_n,
                                position_ids_after_reduction,
                            )

                        hidden_states[:, first_mask_global] = merged.unsqueeze(0)

                        second_mask = second_mask.squeeze()
                        if reduction_stats is not None:
                            reduction_stats[1] += int(second_mask.sum().item())

                        if pc.get_mean_position_id and position_ids_after_reduction is not None:
                            position_ids_after_reduction = position_ids_after_reduction.to(real_position_ids.dtype).permute(
                                2, 1, 0
                            )
                            position_ids[:, :, first_mask_global] = position_ids_after_reduction
                            real_position_ids[:, :, first_mask_global] = position_ids_after_reduction
                    else:
                        second_mask = torch.ones(first_mask_global.sum().item(), dtype=torch.bool, device=device)

                    mask_final = torch.ones(seq_len, dtype=torch.bool, device=device)
                    mask_final[is_image] = first_mask
                    mask_final[first_mask_global] = second_mask

                    position_ids = position_ids[:, :, mask_final]
                    hidden_states = hidden_states[:, mask_final]
                    real_position_ids = real_position_ids[:, :, mask_final]
                    cache_position = cache_position[mask_final]

                    seq_len = hidden_states.shape[1]

                    causal_mask = self._update_causal_mask(
                        attention_mask, hidden_states, cache_position, past_key_values, output_attentions
                    )

                    if pc.get_performance_metrics:
                        torch.cuda.synchronize()
                        _ = time.time() - start_algo

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                # print("index",index,"hidden_states.shape",hidden_states.shape,int(dart_cfg["image_token_length"]))
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs["hidden_states"]

            if use_cache:
                next_decoder_cache = layer_outputs["past_key_value"]
            if output_attentions:
                all_self_attns += (layer_outputs["attn_scores"],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2VLForConditionalGeneration(Qwen2VLFCBase):
    """Same as base ``Qwen2VLForConditionalGeneration`` but LLM uses ``PACT`` instead of ``DART``."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        Qwen2VLPreTrainedModel.__init__(self, config)
        self.visual = DART_ViT._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.model = PACT(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"
        self.post_init()
        self.time_cost_vit = 0
        self.time_cost_llm = 0


# Backwards compatibility for imports of the previous name
Qwen2VLModelPACT = PACT


__all__ = [
    "PACT",
    "PACT_CONFIG",
    "Qwen2VLModelPACT",
    "Qwen2VLForConditionalGeneration",
    "build_pact_namespace",
    "pact_nominal_image_keep_fraction",
    "pact_resolve_final_image_keep_ratio",
    "token_reduction",
    "DBDPC",
]
