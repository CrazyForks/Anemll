"""Qwen 3 model implementation for ANEMLL.

This module provides a lightweight implementation of the Qwen 3 architecture
adapted to the Apple Neural Engine restrictions.  All dense layers are expressed
as ``nn.Conv2d`` with ``kernel_size=1`` and weights are loaded from Hugging Face
checkpoints with the correct reshaping.  Only the pieces required for the unit
 tests are implemented.
"""

from __future__ import annotations

import os
import json
import math
from typing import Dict

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Qwen 3 model implementation adapted from llama_model.py
# ---------------------------------------------------------------------------

MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 512

# LM head configuration constants (following llama_model.py pattern)
ENABLE_CONV2D = bool(1)      # Use Conv2d for LM head
ENABLE_VACAB_SPLIT = bool(1)  # Split vocab into 2 parts
ENABLE_VACAB_SPLIT8 = bool(0)  # Split vocab into 8 parts
ENABLE_VACAB_SPLIT16 = bool(1)  # Split vocab into 16 parts
ENABLE_LOGITS2 = bool(1)    # Return separate logits arrays for CoreML
ENABLE_COREML = bool(0)     # CoreML-specific returns


class QwenConfig:
    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["QwenForCausalLM"])
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.bos_token_id = kwargs.get("bos_token_id", 128000)
        self.eos_token_id = kwargs.get("eos_token_id", 128001)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 14336)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        self.model_type = kwargs.get("model_type", "qwen3")
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.head_dim = kwargs.get(
            "head_dim",
            self.hidden_size // max(1, self.num_attention_heads),
        )
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-05)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling.get("rope_type", "qwen3")
        self.rope_theta = kwargs.get("rope_theta", 500000.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.torch_required = kwargs.get("torch_dtype", "bfloat16")
        self.transformers_version = kwargs.get("transformers_version", "4.40.0.dev0")
        self.use_cache = kwargs.get("use_cache", True)
        self.vocab_size = kwargs.get("vocab_size", 128257)
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", CONTEXT_LENGTH)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# -----------------------------------------------------------------------------
# Qwen building blocks
# -----------------------------------------------------------------------------


class QwenRMSNorm(nn.Module):
    """RMSNorm used in Qwen models - Using true RMSNorm without mean subtraction."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        mean = hidden_states.mean(-1, keepdim=True)
        hidden_states = hidden_states - mean
        return F.layer_norm(hidden_states, self.weight.shape, self.weight, bias=None, eps=float(self.eps)).to(TEST_DEVICE).to(MODEL_DTYPE)

        
        # Use true RMSNorm without mean subtraction (original Qwen implementation)
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(MODEL_DTYPE)  # Ensure consistent output dtype


class QwenHeadNorm(nn.Module):
    """Per-head RMSNorm for query and key projections - Using true RMSNorm without mean subtraction."""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        x=x-mean
        return F.layer_norm(x, self.weight.shape, self.weight, bias=None, eps=float(self.eps)).to(TEST_DEVICE).to(MODEL_DTYPE)


        input_dtype = x.dtype
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(MODEL_DTYPE)  # Ensure consistent output dtype


class QwenRotaryEmbedding(nn.Module):
    """Simple rotary positional embedding."""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.dim, 2).float().to(TEST_DEVICE) / self.dim)
        )
        #inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(TEST_DEVICE) / self.dim))

        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(config.max_position_embeddings, device=TEST_DEVICE).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().unsqueeze(0)
        self.sin_cached = emb.sin().unsqueeze(0)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor | None = None):
        if position_ids is not None:
            # Handle both 1D and 2D position_ids
            if position_ids.dim() == 1:
                pos_ids = position_ids
            else:
                pos_ids = position_ids.squeeze(0)  # Remove batch dimension if present
            
            # Use actual position IDs for correct rotary embeddings
            cos = self.cos_cached[:, pos_ids].to(x.dtype)  # [1, seq_len, head_dim]
            sin = self.sin_cached[:, pos_ids].to(x.dtype)  # [1, seq_len, head_dim]
            return cos, sin
        else:
            # Fallback to sequential positions from 0
            seq_len = x.shape[1]
            return (
                self.cos_cached[:, :seq_len].to(x.dtype),
                self.sin_cached[:, :seq_len].to(x.dtype),
            )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1)
    return hidden_states.view(bsz, n_kv * n_rep, seq_len, head_dim)


class QwenMLP(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Use single Conv2d layers (no splitting for Qwen for now)
        self.gate_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE)
        self.up_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE)
        self.down_proj = nn.Conv2d(self.intermediate_size, self.hidden_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE)

        self.act_fn = F.silu

    def forward(self, x):
        # Use identical step-by-step computation to LlamaMLP to prevent numerical explosion
        x = x.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # Ensure proper dtype and shape
        
        # Step-by-step computation for numerical stability (like LlamaMLP)
        a = self.gate_proj(x)      # gate projection
        b = self.up_proj(x)        # up projection
        c = self.act_fn(a)         # activation on gate
        d = c * b                  # multiply gate * up
        e = self.down_proj(d)      # down projection
        
        return e.squeeze(2).permute(0, 2, 1)  # Final output shape: [bsz, seq_len, hidden_size]


class QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.rotary_emb = QwenRotaryEmbedding(config)

        self.q_proj = nn.Conv2d(
            self.hidden_size,
            self.num_heads * self.head_dim,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.o_proj = nn.Conv2d(
            self.num_heads * self.head_dim,
            self.hidden_size,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.q_norm = QwenHeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = QwenHeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2)
        query_states = (
            self.q_proj(hs)
            .view(bsz, self.num_heads, self.head_dim, seq_len)
            .permute(0, 1, 3, 2)
        )
        key_states = (
            self.k_proj(hs)
            .view(bsz, self.num_kv_heads, self.head_dim, seq_len)
            .permute(0, 1, 3, 2)
        )
        value_states = (
            self.v_proj(hs)
            .view(bsz, self.num_kv_heads, self.head_dim, seq_len)
            .permute(0, 1, 3, 2)
        )

        n_rep = self.num_heads // self.num_kv_heads
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        )
        if causal_mask is not None:
            # Slice causal mask to match seq_len x seq_len for attention weights
            causal_mask_slice = causal_mask[:, :, :seq_len, :seq_len]
            attn_weights = attn_weights + causal_mask_slice.to(attn_weights.dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, -1)
        )
        out = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return out.squeeze(2).permute(0, 2, 1)


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, causal_mask, position_ids, current_pos
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(
            TEST_DEVICE
        )
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the transformer layers."""
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, position_ids, current_pos)
        if IN_PREFILL:
            # Skip final normalization when used only for cache priming
            return hidden_states
        hidden_states = self.norm(hidden_states)
        return hidden_states

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_pretrained_weights(self, model_path: str) -> bool:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(
                    safetensors.torch.load_file(os.path.join(model_path, file))
                )

        conv_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("model.", "") if k.startswith("model.") else k
            if "lm_head.weight" in new_k:
                continue
            if any(
                proj in new_k
                for proj in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                ]
            ):
                conv_state[new_k] = v.view(v.shape[0], v.shape[1], 1, 1)
            else:
                conv_state[new_k] = v

        missing, unexpected = self.load_state_dict(conv_state, strict=False)
        missing = [m for m in missing if "rotary_emb.inv_freq" not in m]
        if missing or unexpected:
            print("Missing keys", missing)
            print("Unexpected keys", unexpected)
        return not missing and not unexpected


class QwenForCausalLM(nn.Module):
    config_class = QwenConfig

    def __init__(self, config: QwenConfig, enable_coreml=False, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.enable_coreml = enable_coreml
        self.model = QwenModel(config)
        
        # Initialize lm_head as Conv2d for ANE optimization following llama_model.py pattern
        if ENABLE_CONV2D:
            if ENABLE_VACAB_SPLIT16:
                vocab_split = config.vocab_size // 16
                vocab_remainder = config.vocab_size % 16
                # Create 16 heads, with the first ones handling any remainder
                for i in range(16):
                    split_size = vocab_split + (1 if i < vocab_remainder else 0)
                    setattr(self, f"lm_head16_{i+1}", 
                           nn.Conv2d(config.hidden_size, split_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE))
                print("Created lm_head16_1 through lm_head16_16")
            elif ENABLE_VACAB_SPLIT8:
                vocab_split = config.vocab_size // 8
                vocab_remainder = config.vocab_size % 8
                # Create 8 heads, with the last one handling any remainder
                for i in range(8):
                    split_size = vocab_split + (1 if i < vocab_remainder else 0)
                    setattr(self, f"lm_head8_{i+1}", 
                           nn.Conv2d(config.hidden_size, split_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE))
                print("Created lm_head8_1 through lm_head8_8")
            elif ENABLE_VACAB_SPLIT:
                self.lm_head2_1 = nn.Conv2d(config.hidden_size, config.vocab_size//2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head2_2 = nn.Conv2d(config.hidden_size, config.vocab_size//2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                print("Created lm_head2_1 and lm_head2_2")
            else:
                self.lm_head1 = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                print("Created lm_head1")
        else:
            # Use linear head
            self.lm_head = nn.Conv2d(
                config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE
            ).to(TEST_DEVICE)
            print("Created linear lm_head")

    def forward(
        self,
        input_ids: torch.LongTensor,
        update_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        current_pos: torch.LongTensor,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        assert len(input_ids.shape) == 2, "input_ids must be 2D"
        if not IN_PREFILL:
            assert position_ids.ndim in (1, 2), "position_ids must be 1D or 2D"
        else:
            assert (
                position_ids.shape[-1] == input_ids.shape[-1]
            ), "position_ids length must match input_ids in prefill"

        hidden_states = self.model(
            input_ids,
            causal_mask,
            position_ids,
            current_pos,
            IN_PREFILL=IN_PREFILL,
        )
        
        # Extract hidden states at current position right before LM head (using current_pos dynamically)
        if not IN_PREFILL and current_pos is not None:
            # Use torch.index_select for dynamic position extraction that traces well
            if isinstance(current_pos, torch.Tensor):
                pos_tensor = current_pos if current_pos.dim() > 0 else current_pos.unsqueeze(0)
            else:
                pos_tensor = torch.tensor([current_pos], device=hidden_states.device, dtype=torch.long)
            
            # Use index_select which should create proper dynamic slicing in CoreML using current_pos
            hidden_states = torch.index_select(hidden_states, dim=1, index=pos_tensor)  # [batch, 1, hidden_size]
        
        # Project to vocabulary using appropriate head
        if ENABLE_CONV2D:
            # Reshape for Conv2d and ensure float16
            hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
            
            if ENABLE_VACAB_SPLIT16:
                # Use 16-way split head
                logits1 = self.lm_head16_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head16_2(hidden_states).squeeze(2).transpose(1, 2)
                logits3 = self.lm_head16_3(hidden_states).squeeze(2).transpose(1, 2)
                logits4 = self.lm_head16_4(hidden_states).squeeze(2).transpose(1, 2)
                logits5 = self.lm_head16_5(hidden_states).squeeze(2).transpose(1, 2)
                logits6 = self.lm_head16_6(hidden_states).squeeze(2).transpose(1, 2)
                logits7 = self.lm_head16_7(hidden_states).squeeze(2).transpose(1, 2)
                logits8 = self.lm_head16_8(hidden_states).squeeze(2).transpose(1, 2)
                logits9 = self.lm_head16_9(hidden_states).squeeze(2).transpose(1, 2)
                logits10 = self.lm_head16_10(hidden_states).squeeze(2).transpose(1, 2)
                logits11 = self.lm_head16_11(hidden_states).squeeze(2).transpose(1, 2)
                logits12 = self.lm_head16_12(hidden_states).squeeze(2).transpose(1, 2)
                logits13 = self.lm_head16_13(hidden_states).squeeze(2).transpose(1, 2)
                logits14 = self.lm_head16_14(hidden_states).squeeze(2).transpose(1, 2)
                logits15 = self.lm_head16_15(hidden_states).squeeze(2).transpose(1, 2)
                logits16 = self.lm_head16_16(hidden_states).squeeze(2).transpose(1, 2)
                
                if self.enable_coreml and ENABLE_LOGITS2:
                    return logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8, logits9, logits10, logits11, logits12, logits13, logits14, logits15, logits16
                else:
                    logits = torch.cat([logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8, logits9, logits10, logits11, logits12, logits13, logits14, logits15, logits16], dim=2)
            
            elif ENABLE_VACAB_SPLIT8:
                # Use 8-way split head
                logits1 = self.lm_head8_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head8_2(hidden_states).squeeze(2).transpose(1, 2)
                logits3 = self.lm_head8_3(hidden_states).squeeze(2).transpose(1, 2)
                logits4 = self.lm_head8_4(hidden_states).squeeze(2).transpose(1, 2)
                logits5 = self.lm_head8_5(hidden_states).squeeze(2).transpose(1, 2)
                logits6 = self.lm_head8_6(hidden_states).squeeze(2).transpose(1, 2)
                logits7 = self.lm_head8_7(hidden_states).squeeze(2).transpose(1, 2)
                logits8 = self.lm_head8_8(hidden_states).squeeze(2).transpose(1, 2)
                
                if self.enable_coreml and ENABLE_LOGITS2:
                    return logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8
                else:
                    logits = torch.cat([logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8], dim=2)
            
            elif ENABLE_VACAB_SPLIT:
                # Use 2-way split head
                logits1 = self.lm_head2_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head2_2(hidden_states).squeeze(2).transpose(1, 2)
                
                if self.enable_coreml and ENABLE_LOGITS2:
                    return logits1, logits2
                
                logits = torch.cat([logits1, logits2], dim=2)
            
            else:
                # Use single head
                logits = self.lm_head1(hidden_states).squeeze(2).transpose(1, 2)
        else:
            # Use linear head (fallback)
            logits = self.lm_head(hidden_states.permute(0, 2, 1).unsqueeze(2))
            logits = logits.squeeze(2).permute(0, 2, 1)
        
        return logits

    def prefill_kv_cache(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        start_pos: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> None:
        seq_len = input_ids.shape[1]
        causal_slice = (
            causal_mask[:, :, :seq_len, :] if causal_mask is not None else None
        )
        with torch.no_grad():
            self.model(
                input_ids,
                causal_slice,
                position_ids,
                start_pos,
                IN_PREFILL=True,
            )

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not self.model.load_pretrained_weights(model_path):
            return False
        
        # Load lm_head weights with splitting support
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(
                    safetensors.torch.load_file(os.path.join(model_path, file))
                )
        
        # Handle lm_head weight loading and splitting
        lm_head_weight = None
        for k, v in state_dict.items():
            if k == "lm_head.weight":
                lm_head_weight = v
                break
        
        if lm_head_weight is not None:
            if ENABLE_CONV2D:
                reshaped_weight = lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1)
                if ENABLE_VACAB_SPLIT16:
                    vocab_split = self.config.vocab_size // 16
                    vocab_remainder = self.config.vocab_size % 16
                    # Create splits with proper sizes, distributing remainder among first splits
                    split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]
                    splits = torch.split(reshaped_weight, split_sizes)
                    for i, split in enumerate(splits):
                        getattr(self, f"lm_head16_{i+1}").weight.data.copy_(split)
                        print(f"Loaded lm_head16_{i+1}.weight with shape {split.shape}")
                elif ENABLE_VACAB_SPLIT8:
                    vocab_split = self.config.vocab_size // 8
                    vocab_remainder = self.config.vocab_size % 8
                    # Create splits with proper sizes, distributing remainder among first splits
                    split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(8)]
                    splits = torch.split(reshaped_weight, split_sizes)
                    for i, split in enumerate(splits):
                        getattr(self, f"lm_head8_{i+1}").weight.data.copy_(split)
                        print(f"Loaded lm_head8_{i+1}.weight with shape {split.shape}")
                elif ENABLE_VACAB_SPLIT:
                    vocab_split = self.config.vocab_size // 2
                    split1, split2 = torch.split(reshaped_weight, [vocab_split, self.config.vocab_size - vocab_split])
                    self.lm_head2_1.weight.data.copy_(split1)
                    self.lm_head2_2.weight.data.copy_(split2)
                    print(f"Loaded lm_head2_1.weight and lm_head2_2.weight")
                else:
                    self.lm_head1.weight.data.copy_(reshaped_weight)
                    print(f"Loaded lm_head1.weight")
            else:
                self.lm_head.weight.data.copy_(lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1))
        else:
            print("Warning: lm_head.weight not found in model weights")
            return False
        
        return True
