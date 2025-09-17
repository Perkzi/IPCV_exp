import torch
import torch.nn as nn

class KVFlopsMeter:
    def __init__(self, model):
        self.model = model
        self.total_flops = 0
        self.total_kv_MB = 0
        self.sample_count = 0
        self.hooks = []

    def _attn_hook(self, module, inputs, output):
        # FLOPs
        print("input",inputs,inputs[0].shape)
        x = inputs[0]
        print(x.shape,len(x.shape))
        if not isinstance(x, torch.Tensor):
            return
        if len(x.shape)==3:
            B, Lq, D = x.shape
        else:
            B = 1
            Lq, D = x.shape

        num_heads = getattr(module, "num_heads", None) or getattr(module, "n_heads", None)
        head_dim = getattr(module, "head_dim", None) or (D // num_heads if num_heads else None)
        if num_heads is None or head_dim is None:
            return

        # KV 长度
        Lkv = Lq
        past_kv = None
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            past_kv = output[2]
        if past_kv is not None and isinstance(past_kv, (tuple, list)) and len(past_kv) == 2:
            k, v = past_kv
            if isinstance(k, torch.Tensor) and k.dim() >= 3:
                Lkv = k.shape[2]
                # KV Cache 只统计一次（prefill）
                if self._counting_kv:
                    kv_bytes = (k.numel() + v.numel()) * k.element_size()
                    self.total_kv_MB += kv_bytes / (1024**2)
                    self._counting_kv = False

        attn_flops = 2 * B * num_heads * Lq * Lkv * head_dim
        self.total_flops += attn_flops

    def _mlp_hook(self, module, inputs, output):
        print("input",inputs,inputs[0].shape)
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return
        
        if len(x.shape)==3:
            B, N, Din = x.shape
        else:
            B = 1
            N, Din = x.shape

        if hasattr(module, "fc1") and hasattr(module, "fc2"):
            Dout = module.fc1.out_features
            self.total_flops += 2 * B * N * Din * Dout
            self.total_flops += 2 * B * N * Dout * Din

    def start(self):
        # 注册 hook
        for name, m in self.model.named_modules():
            if ("model.layers" in name and "self_attn" in name) or ("visual.blocks" in name and "attn" in name):
                self.hooks.append(m.register_forward_hook(self._attn_hook))
            if ("model.layers" in name and "mlp" in name) or ("visual.blocks" in name and "mlp" in name):
                self.hooks.append(m.register_forward_hook(self._mlp_hook))

    def stop(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def record_sample(self):
        self.sample_count += 1
        self._counting_kv = True  # 每个样本开始时允许统计一次 KV

    def get_results(self):
        avg_flops = self.total_flops / self.sample_count if self.sample_count else 0
        avg_kv_MB = self.total_kv_MB / self.sample_count if self.sample_count else 0
        return avg_flops, avg_kv_MB