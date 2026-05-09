import math
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import torch.nn.functional as F


class ChaokaiMindConfig(PretrainedConfig):
    model_type = "chaokaimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get(
            "moe_intermediate_size", self.intermediate_size
        )
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x: torch.Tensor):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    def forward(self, x: torch.Tensor):
        return x * self._norm(x.float()).type_as(x) * self.weight
    
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


def repeat_kv(x: torch.Tensor, num_kv_repeats: int):
    if num_kv_repeats == 1:
        return x
    batch_size, seq_len, num_key_value_heads ,dim = x.shape
    return x[:, :,: ,None, :].expand(batch_size, seq_len, num_key_value_heads ,num_kv_repeats, dim).reshape(batch_size, seq_len, num_key_value_heads * num_kv_repeats, dim)

class Attention(torch.nn.Module):
    def __init__(self, config: ChaokaiMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.is_casual = True
        self.num_kv_repeats = self.num_attention_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_droupout = nn.Dropout(config.dropout)
        self.residual_droupout = nn.Dropout(config.dropout)
        self.flash = config.flash_attn and hasattr(torch.nn.functional, "scaled_dot_product_attention")
    def forward(self, x, position_embeddings, past_kv = None, use_cache=False, attn_mask=None):
        batch_size, seq_len, _ = x.size()
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_kv is not None:
            xk = torch.cat([past_kv[0], xk], dim=1)
            xv = torch.cat([past_kv[1], xv], dim=1)
        if use_cache:
            past_kv = (xk, xv)
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.num_kv_repeats).transpose(1, 2)
        xv = repeat_kv(xv, self.num_kv_repeats).transpose(1, 2)

        if self.flash and (seq_len > 1) and (not self.is_casual or past_kv is None):
            attn_output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.attn_droupout if self.training else 0.0)
        else:
            attn_scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
            if self.is_casual:
                attn_scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float('-inf'), device=attn_scores.device).triu(diagonal=1)
            if attn_mask is not None:
                attn_scores += (1 - attn_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            attn_output = self.attn_droupout(F.softmax(attn_scores.float(), dim=-1)).type_as(xq) @ xv
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.attn_droupout(self.out_proj(attn_output))
        return attn_output, past_kv








        





