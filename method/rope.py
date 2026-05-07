import math
import torch

def precompute_angles(dim, end, base=1e4, rope_scaling=None):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    attn_factor = rope_scaling.get("attention_factor", 1.0) if rope_scaling else 1.0
    if rope_scaling and rope_scaling["type"] == "yarn":
        beta_fast=rope_scaling.get("beta_fast", 32)
        beta_slow=rope_scaling.get("beta_slow", 1)
        factor=rope_scaling.get("factor", 16) 
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 2048)
        if end > original_max_position_embeddings:
             def inv_dim(beta):
                 return (dim * math.log(original_max_position_embeddings / (2 * math.pi * beta))) /(2 * math.log(base))
             low = max(math.floor(inv_dim(beta_fast)), 0)
             high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
             ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
             freqs = freqs * (1 -ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    angles = torch.outer(t, freqs).float()
    angles_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attn_factor
    angles_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attn_factor
    return angles_cos, angles_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    q_rotated = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_rotated = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    return q_rotated, k_rotated    

