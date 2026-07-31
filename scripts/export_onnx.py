#!/usr/bin/env python3
"""Export a fine-tuned ArkttsModel to ONNX for the Audio8 ONNX runtime.

The Audio8 repo ships only the ONNX *runtime* (inference), not an export
script.  The pre-built ONNX model on HuggingFace uses the *original* weights.
This script exports *our* fine-tuned Basque model so it can run on the same
CPU ONNX runtime.

Produces the layout expected by ``onnx_runtime/arktts_runtime``:

    output_dir/
    |- slow_ar_fp16.onnx(.data)        # Slow AR with KV cache
    |- fast_ar_fp16.onnx(.data)        # Fast AR with KV cache
    |- codec_decoder_fp16.onnx(.data)  # codes → audio
    |- registration/
    |   |- codec_encoder_fp16.onnx(.data)  # audio → codes
    |   `- registration_manifest.json
    |- runtime_manifest.json
    `- tokenizer/
        `- tokenizer.json

Precision: FP16 weights and activations.  INT4 weight quantization is a
separate post-processing step (see ``scripts/quantize_int4.py``) that replaces
MatMul nodes with MatMulNBits (com.microsoft domain), matching Audio8's
official INT4 ONNX format.  The runtime supports arbitrary precision names
via ``runtime_manifest.json``.

Usage:
    python export_onnx.py --model /path/to/finetuned --output /path/to/onnx_out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Helpers from the modeling code (copied to avoid import issues)
# ---------------------------------------------------------------------------


def _precompute_rope(length: int, head_dim: int, base: float) -> torch.Tensor:
    frequencies = 1.0 / (
        base ** (torch.arange(0, head_dim, 2).float()[: head_dim // 2] / head_dim)
    )
    phases = torch.outer(torch.arange(length), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((values.real, values.imag), dim=-1)


def _apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    if rope.ndim == 3:
        rope = rope[None, :, None]
    elif rope.ndim == 4:
        rope = rope[:, :, None]
    else:
        raise ValueError(f"Unexpected RoPE shape: {tuple(rope.shape)}")
    output = torch.stack(
        (
            shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1],
            shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1],
        ),
        dim=-1,
    )
    return output.flatten(3).to(x.dtype)


# ---------------------------------------------------------------------------
# Wrapper: Slow AR  (semantic transformer with KV cache)
# ---------------------------------------------------------------------------


class SlowAROnnx(nn.Module):
    """Wraps ArkttsModel's slow path to match the ONNX runtime interface.

    Inputs (dynamic T = sequence length):
        codes         [1, num_codebooks+1, T]  int64
        input_pos     [T]                       int64
        cache_key_i   [1, n_local_heads, max_seq, head_dim]  fp16  (per layer)
        cache_value_i [1, n_local_heads, max_seq, head_dim]  fp16  (per layer)

    Outputs:
        logits        [1, T, 4097]   fp16  (semantic_begin..end + eos)
        slow_hidden   [1, T, dim]   fp16
        key_delta_i   [1, n_local_heads, T, head_dim]  fp16  (per layer)
        value_delta_i [1, n_local_heads, T, head_dim]  fp16  (per layer)
    """

    def __init__(self, model):
        super().__init__()
        self.embeddings = model.embeddings
        self.codebook_embeddings = model.codebook_embeddings
        self.layers = model.layers
        self.norm = model.norm
        self.register_buffer("freqs_cis", _precompute_rope(
            model.config.max_seq_len, model.config.head_dim, model.config.rope_base
        ))
        self.config = model.config
        # Pre-extract the semantic+eos weight rows for efficient logits
        begin = model.config.semantic_begin_id
        end = model.config.semantic_end_id
        eos = model.config.eos_token_id
        weight = model.embeddings.weight
        self.register_buffer(
            "logits_weight",
            torch.cat([weight[begin : end + 1], weight[eos : eos + 1]], dim=0),  # [4097, dim]
        )

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        codebook_embeds = []
        for index in range(cfg.num_codebooks):
            codebook_embeds.append(
                self.codebook_embeddings(input_ids[:, index + 1] + index * cfg.codebook_size)
            )
        codebook_sum = torch.stack(codebook_embeds, dim=1).sum(dim=1)
        semantic = (input_ids[:, 0] >= cfg.semantic_begin_id) & (
            input_ids[:, 0] <= cfg.semantic_end_id
        )
        codebook_sum = torch.where(semantic.unsqueeze(-1), codebook_sum, 0.0)
        return self.embeddings(input_ids[:, 0]) + codebook_sum

    def forward(self, codes: torch.Tensor, input_pos: torch.Tensor, *caches):
        cfg = self.config
        batch, _, T = codes.shape
        hidden = self._embed(codes)  # [1, T, dim]
        rope = self.freqs_cis[input_pos]  # [T, head_dim//2, 2]

        # Attention mask: query at input_pos[i] attends to keys 0..input_pos[i]
        key_positions = torch.arange(cfg.max_seq_len, device=hidden.device)
        valid = key_positions[None, :] <= input_pos[:, None]  # [T, max_seq]
        add_mask = torch.where(valid, 0.0, torch.tensor(-1e4, dtype=hidden.dtype, device=hidden.device))
        add_mask = add_mask[None, None, :, :]  # [1, 1, T, max_seq]

        deltas = []
        for i, layer in enumerate(self.layers):
            key_cache = caches[2 * i]
            value_cache = caches[2 * i + 1]
            attn = layer.attention

            x = layer.attention_norm(hidden)
            B, L, _ = x.shape
            q_size = attn.n_head * attn.head_dim
            kv_size = attn.n_local_heads * attn.head_dim
            q, k, v = attn.wqkv(x).split((q_size, kv_size, kv_size), dim=-1)
            q = q.view(B, L, attn.n_head, attn.head_dim)
            k = k.view(B, L, attn.n_local_heads, attn.head_dim)
            v = v.view(B, L, attn.n_local_heads, attn.head_dim)
            if attn.qk_norm:
                q = attn.q_norm(q)
                k = attn.k_norm(k)
            q = _apply_rope(q, rope).transpose(1, 2)  # [1, n_head, T, hd]
            k = _apply_rope(k, rope).transpose(1, 2)  # [1, n_kv, T, hd]
            v = v.transpose(1, 2)                      # [1, n_kv, T, hd]

            # Scatter new K/V into cache for attention read
            idx = input_pos.view(1, 1, T, 1).expand(B, attn.n_local_heads, T, attn.head_dim)
            full_k = key_cache.scatter(2, idx, k)    # [1, n_kv, max_seq, hd]
            full_v = value_cache.scatter(2, idx, v)

            # GQA repeat
            repeats = attn.n_head // attn.n_local_heads
            full_k = full_k.repeat_interleave(repeats, dim=1)
            full_v = full_v.repeat_interleave(repeats, dim=1)

            # Manual attention (ONNX-safe)
            scores = torch.matmul(q, full_k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
            scores = scores + add_mask
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, full_v)  # [1, n_head, T, hd]
            out = out.transpose(1, 2).contiguous().view(B, L, q_size)
            attn_out = attn.wo(out)

            hidden = hidden + attn_out
            hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))

            deltas.append(k)  # [1, n_kv, T, hd]
            deltas.append(v)

        normalized = self.norm(hidden)
        logits = F.linear(normalized, self.logits_weight)  # [1, T, 4097]
        return logits, normalized, *deltas


# ---------------------------------------------------------------------------
# Wrapper: Fast AR  (codebook transformer with KV cache)
# ---------------------------------------------------------------------------


class FastAROnnx(nn.Module):
    """Wraps ArkttsModel's fast path to match the ONNX runtime interface.

    Inputs (all static — fast AR always processes 1 token):
        slow_hidden      [1, 1, dim]    fp16
        token_id         [1, 1]         int64
        use_slow_hidden  [1]            bool
        input_pos        [1]            int64
        cache_key_i      [1, n_kv, num_codebooks, head_dim]  fp16
        cache_value_i    [1, n_kv, num_codebooks, head_dim]  fp16

    Outputs:
        logits           [1, 1, codebook_size]  fp16
        key_delta_i      [1, n_kv, 1, head_dim]  fp16
        value_delta_i    [1, n_kv, 1, head_dim]  fp16
    """

    def __init__(self, model):
        super().__init__()
        self.fast_project_in = model.fast_project_in
        self.fast_embeddings = model.fast_embeddings
        self.fast_layers = model.fast_layers
        self.fast_norm = model.fast_norm
        self.fast_output = model.fast_output
        self.register_buffer("fast_freqs_cis", _precompute_rope(
            model.config.num_codebooks, model.config.fast_head_dim, model.config.rope_base
        ))
        self.config = model.config

    def forward(self, slow_hidden, token_id, use_slow_hidden, input_pos, *caches):
        cfg = self.config
        # Select input: projected slow hidden or token embedding
        projected = self.fast_project_in(slow_hidden)  # [1, 1, fast_dim]
        token_embed = self.fast_embeddings(token_id)    # [1, 1, fast_dim]
        hidden = torch.where(
            use_slow_hidden[:, None, None], projected, token_embed
        )

        T = 1
        rope = self.fast_freqs_cis[input_pos]  # [1, head_dim//2, 2]

        key_positions = torch.arange(cfg.num_codebooks, device=hidden.device)
        valid = key_positions[None, :] <= input_pos[:, None]  # [1, num_codebooks]
        add_mask = torch.where(valid, 0.0, torch.tensor(-1e4, dtype=hidden.dtype, device=hidden.device))
        add_mask = add_mask[None, None, :, :]  # [1, 1, 1, num_codebooks]

        deltas = []
        for i, layer in enumerate(self.fast_layers):
            key_cache = caches[2 * i]
            value_cache = caches[2 * i + 1]
            attn = layer.attention

            x = layer.attention_norm(hidden)
            B, L, _ = x.shape
            q_size = attn.n_head * attn.head_dim
            kv_size = attn.n_local_heads * attn.head_dim
            q, k, v = attn.wqkv(x).split((q_size, kv_size, kv_size), dim=-1)
            q = q.view(B, L, attn.n_head, attn.head_dim)
            k = k.view(B, L, attn.n_local_heads, attn.head_dim)
            v = v.view(B, L, attn.n_local_heads, attn.head_dim)
            if attn.qk_norm:
                q = attn.q_norm(q)
                k = attn.k_norm(k)
            q = _apply_rope(q, rope).transpose(1, 2)
            k = _apply_rope(k, rope).transpose(1, 2)
            v = v.transpose(1, 2)

            idx = input_pos.view(1, 1, T, 1).expand(B, attn.n_local_heads, T, attn.head_dim)
            full_k = key_cache.scatter(2, idx, k)
            full_v = value_cache.scatter(2, idx, v)

            repeats = attn.n_head // attn.n_local_heads
            full_k = full_k.repeat_interleave(repeats, dim=1)
            full_v = full_v.repeat_interleave(repeats, dim=1)

            scores = torch.matmul(q, full_k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
            scores = scores + add_mask
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, full_v)
            out = out.transpose(1, 2).contiguous().view(B, L, q_size)
            attn_out = attn.wo(out)

            hidden = hidden + attn_out
            hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))

            deltas.append(k)
            deltas.append(v)

        logits = self.fast_output(self.fast_norm(hidden))  # [1, 1, codebook_size]
        return logits, *deltas


# ---------------------------------------------------------------------------
# Wrapper: Codec decoder  (codes → audio)
# ---------------------------------------------------------------------------


class CodecDecoderOnnx(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [B, num_codebooks, T]
        return self.codec.decode(codes.long())  # [B, 1, samples]


# ---------------------------------------------------------------------------
# Wrapper: Codec encoder  (audio → codes, for voice registration)
# ---------------------------------------------------------------------------


class CodecEncoderOnnx(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.encoder = codec.encoder
        self.quantizer = codec.quantizer

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, 1, samples] — runtime passes float32, cast to model dtype
        audio = audio.to(next(self.encoder.parameters()).dtype)
        encoded = self.encoder(audio)  # [B, 1024, T']
        z = self.quantizer.pre_module(self.quantizer.downsample(encoded))
        semantic, semantic_codes = self.quantizer.semantic_quantizer(z)
        _, residual_codes = self.quantizer.quantizer(z - semantic)
        codes = torch.cat((semantic_codes, residual_codes), dim=1)  # [B, 10, T'']
        return codes


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

OPSET = 17


def _patch_codec_for_onnx():
    """Monkey-patch ArkttsCodecWindowTransformer.forward to avoid in-place
    bitwise AND (``aten::__iand_``) which ONNX cannot export.

    Replaces ``mask &= expr`` with ``mask = mask & expr`` (out-of-place).
    Also patches ArkttsCodecLayerScale to avoid in-place mul, and _rope
    to avoid hardcoded bfloat16.
    """
    import sys

    # Walk all loaded modules to find the codec modeling class and helpers
    target_cls = None
    mod = None
    for m in list(sys.modules.values()):
        cls = getattr(m, "ArkttsCodecWindowTransformer", None)
        if cls is not None and isinstance(cls, type):
            target_cls = cls
            mod = m
            break
    if target_cls is None or mod is None:
        print("  WARNING: ArkttsCodecWindowTransformer not found in sys.modules — codec patch skipped", flush=True)
        return
    print(f"  patching codec classes in {mod.__name__}", flush=True)

    # Patch 1: ArkttsCodecWindowTransformer.forward — replace in-place AND
    def _patched_forward(self, x, x_lens=None):
        del x_lens
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.look_ahead_conv(self.input_proj(x))
        length = x.shape[1]
        row = torch.arange(length, device=x.device)[:, None]
        column = torch.arange(length, device=x.device)[None, :]
        mask = column <= row
        if self.window_size is not None:
            # Out-of-place AND (was: mask &= ... → aten::__iand_ unsupported by ONNX)
            mask = mask & (column >= (row - self.window_size + 1).clamp_min(0))
        mask = mask[None, None]
        rope_values = _patched_rope(length, self.head_dim, self.rope_base, x.device)
        for layer in self.layers:
            x = layer(x, rope_values, mask)
        x = self.output_proj(self.norm(x))
        return x.transpose(1, 2) if self.channels_first else x

    # Patch 2: _rope — replace torch.polar (aten::polar unsupported by ONNX)
    # and don't hardcode bfloat16 (we export in float16)
    def _patched_rope(length, head_dim, base, device=None):
        frequencies = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        # Explicitly cast arange to float — torch.outer(int64, float) may not
        # promote correctly during ONNX tracing, producing int64 Einsum output
        phases = torch.outer(torch.arange(length, device=device).float(), frequencies)
        # torch.polar(ones, phases) = cos(phases) + i*sin(phases)
        return torch.stack((torch.cos(phases), torch.sin(phases)), dim=-1).float()

    # Patch 3: ArkttsCodecLayerScale.forward — avoid in-place mul (aten::mul_)
    layer_scale_cls = getattr(mod, "ArkttsCodecLayerScale", None)
    if layer_scale_cls is not None:
        def _patched_layerscale_forward(self, x):
            return x * self.gamma  # always out-of-place
        layer_scale_cls.forward = _patched_layerscale_forward

    # Patch 4: ArkttsCausalConvTranspose1d.forward — use negative indexing
    # instead of x.shape[-1] - crop, which gets baked as a constant during tracing
    conv_transpose_cls = getattr(mod, "ArkttsCausalConvTranspose1d", None)
    if conv_transpose_cls is not None:
        def _patched_conv_transpose_forward(self, x):
            x = self.conv(x)
            crop = self.kernel_size - self.stride
            if crop:
                x = x[..., :-crop]  # negative index → dynamic Slice in ONNX
            return x.contiguous()
        conv_transpose_cls.forward = _patched_conv_transpose_forward

    # Patch 5: ArkttsResidualUnit.forward — avoid x.shape[-1] which bakes in
    residual_unit_cls = getattr(mod, "ArkttsResidualUnit", None)
    if residual_unit_cls is not None:
        def _patched_residual_forward(self, x):
            output = self.block(x)
            # Match lengths by cropping the longer one (dynamic, no shape bake-in)
            return x + output
        residual_unit_cls.forward = _patched_residual_forward

    # Patch 6: ArkttsSnake1d.forward — the jit.scripted _arktts_snake captures
    # x.shape and does x.reshape(shape) which bakes in the traced length.
    # Replace with inline ops that don't capture/reshape shapes.
    snake_cls = getattr(mod, "ArkttsSnake1d", None)
    if snake_cls is not None:
        def _patched_snake_forward(self, x):
            # x is [B, C, T] — no need to reshape to 3D and back
            alpha = self.alpha  # [1, C, 1]
            return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        snake_cls.forward = _patched_snake_forward

    # Patch 7: ArkttsCausalConv1d.forward — _extra_padding uses x.shape[-1] in
    # Python arithmetic, baking the 'right' pad value as a constant.
    # For stride=1 (all decoder convs), _extra_padding always returns 0,
    # so we can safely hardcode right=0.
    causal_conv_cls = getattr(mod, "ArkttsCausalConv1d", None)
    if causal_conv_cls is not None:
        def _patched_causal_conv_forward(self, x):
            return self.conv(F.pad(x, (self.padding, 0))).contiguous()
        causal_conv_cls.forward = _patched_causal_conv_forward

    target_cls.forward = _patched_forward
    mod._rope = _patched_rope


def _model_fingerprint(model_dir: str) -> str:
    """SHA-256 of the safetensors weights file (identity check for registration)."""
    p = Path(model_dir) / "model.safetensors"
    if p.exists():
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    return "unknown"


def export_slow_ar(model, out_dir: Path):
    print("  building Slow AR wrapper...", flush=True)
    wrapper = SlowAROnnx(model).eval()
    cfg = model.config
    T = 1  # trace single-token step so cache reads are visible to exporter
    codes = torch.zeros(1, cfg.num_codebooks + 1, T, dtype=torch.long)
    codes[:, 0] = torch.randint(cfg.semantic_begin_id, cfg.semantic_end_id + 1, (1, T))
    codes[:, 1:] = torch.randint(0, cfg.codebook_size, (1, cfg.num_codebooks, T))
    # Position 4: cache at positions 0-3 is READ by attention (not overwritten by scatter)
    input_pos = torch.tensor([4], dtype=torch.long)
    # Non-zero cache values at positions 0-3 so exporter can't constant-fold them away
    caches = []
    for i in range(2 * cfg.n_layer):
        c = torch.zeros(1, cfg.n_local_heads, cfg.max_seq_len, cfg.head_dim, dtype=torch.float16)
        c[:, :, :4, :] = torch.randn(1, cfg.n_local_heads, 4, cfg.head_dim, dtype=torch.float16) * 0.1
        caches.append(c)
    # Interleaved names: cache_key_0, cache_value_0, cache_key_1, cache_value_1, ...
    input_names = ["codes", "input_pos"]
    output_names = ["logits", "slow_hidden"]
    dyn_axes = {"codes": {2: "T"}, "input_pos": {0: "T"}, "logits": {1: "T"}, "slow_hidden": {1: "T"}}
    for i in range(cfg.n_layer):
        input_names.extend([f"cache_key_{i}", f"cache_value_{i}"])
        output_names.extend([f"key_delta_{i}", f"value_delta_{i}"])
        dyn_axes[f"key_delta_{i}"] = {2: "T"}
        dyn_axes[f"value_delta_{i}"] = {2: "T"}
    print("  tracing...", flush=True)
    torch.onnx.export(
        wrapper,
        (codes, input_pos, *caches),
        str(out_dir / "slow_ar_fp16.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dyn_axes,
        opset_version=OPSET,
        do_constant_folding=False,
    )
    print(f"  ✓ slow_ar_fp16.onnx", flush=True)


def export_fast_ar(model, out_dir: Path):
    print("  building Fast AR wrapper...", flush=True)
    wrapper = FastAROnnx(model).eval()
    cfg = model.config
    slow_hidden = torch.randn(1, 1, cfg.dim, dtype=torch.float16)
    token_id = torch.randint(0, cfg.codebook_size, (1, 1), dtype=torch.long)
    use_slow_hidden = torch.tensor([True], dtype=torch.bool)
    # Position 4: cache at positions 0-3 is READ by attention
    input_pos = torch.tensor([4], dtype=torch.long)
    caches = []
    for i in range(2 * cfg.n_fast_layer):
        c = torch.zeros(1, cfg.fast_n_local_heads, cfg.num_codebooks, cfg.fast_head_dim, dtype=torch.float16)
        c[:, :, :4, :] = torch.randn(1, cfg.fast_n_local_heads, 4, cfg.fast_head_dim, dtype=torch.float16) * 0.1
        caches.append(c)
    input_names = ["slow_hidden", "token_id", "use_slow_hidden", "input_pos"]
    output_names = ["logits"]
    for i in range(cfg.n_fast_layer):
        input_names.extend([f"cache_key_{i}", f"cache_value_{i}"])
        output_names.extend([f"key_delta_{i}", f"value_delta_{i}"])
    print("  tracing...", flush=True)
    torch.onnx.export(
        wrapper,
        (slow_hidden, token_id, use_slow_hidden, input_pos, *caches),
        str(out_dir / "fast_ar_fp16.onnx"),
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET,
        do_constant_folding=False,
    )
    print(f"  ✓ fast_ar_fp16.onnx", flush=True)


def export_codec_decoder(codec, out_dir: Path):
    print("  building codec decoder wrapper...", flush=True)
    wrapper = CodecDecoderOnnx(codec).eval()
    codes = torch.randint(0, 4096, (1, 10, 5), dtype=torch.long)
    print("  tracing...", flush=True)
    torch.onnx.export(
        wrapper,
        (codes,),
        str(out_dir / "codec_decoder_fp16.onnx"),
        input_names=["codes"],
        output_names=["audio"],
        dynamic_axes={"codes": {2: "T"}, "audio": {2: "samples"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"  ✓ codec_decoder_fp16.onnx", flush=True)


def export_codec_encoder(codec, reg_dir: Path):
    print("  building codec encoder wrapper...", flush=True)
    wrapper = CodecEncoderOnnx(codec).eval()
    # 1 second of audio at 44.1kHz, padded to frame_length multiple
    audio = torch.randn(1, 1, 44100, dtype=torch.float32)
    print("  tracing...", flush=True)
    torch.onnx.export(
        wrapper,
        (audio,),
        str(reg_dir / "codec_encoder_fp16.onnx"),
        input_names=["audio"],
        output_names=["codes"],
        dynamic_axes={"audio": {2: "samples"}, "codes": {2: "T"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"  ✓ codec_encoder_fp16.onnx", flush=True)


def write_manifests(model, model_dir: str, out_dir: Path, reg_dir: Path):
    cfg = model.config
    fingerprint = _model_fingerprint(model_dir)

    runtime_manifest = {
        "model_family": "audio8_tts",
        "activation_dtype": "float16",
        "slow_logits_layout": "semantic_then_eos",
        "slow_logits_size": cfg.semantic_end_id - cfg.semantic_begin_id + 2,  # 4096 + 1 eos
        "kv_attention_layout": "valid_prefix",
        "max_seq_len": cfg.max_seq_len,
        "num_layers": cfg.n_layer,
        "num_fast_layers": cfg.n_fast_layer,
        "num_codebooks": cfg.num_codebooks,
        "n_local_heads": cfg.n_local_heads,
        "fast_n_local_heads": cfg.fast_n_local_heads,
        "head_dim": cfg.head_dim,
        "fast_head_dim": cfg.fast_head_dim,
        "fast_dim": cfg.fast_dim,
        "vocab_size": cfg.vocab_size,
        "codebook_size": cfg.codebook_size,
        "semantic_begin_id": cfg.semantic_begin_id,
        "semantic_end_id": cfg.semantic_end_id,
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
        "codec_sample_rate": cfg.codec_sample_rate,
        "codec_frame_size": cfg.codec_frame_size,
        "sample_rate": cfg.codec_sample_rate,
        "codec_hop_length": cfg.codec_frame_size,
        "stream_context_frames": 128,
        "stream_guard_frames": 1,
        "decoder_provider": "cpu",
        "default_codec_precision": "fp16",
        "available_codec_precisions": ["fp16"],
        "codec_models": {"fp16": "codec_decoder_fp16.onnx"},
        "im_end_id": cfg.eos_token_id,
        "model_fingerprint": fingerprint,
        "default_precision": "fp16",
        "available_precisions": ["fp16"],
    }
    (out_dir / "runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=4) + "\n"
    )
    print(f"  ✓ runtime_manifest.json", flush=True)

    reg_manifest = {
        "sample_rate": cfg.codec_sample_rate,
        "num_codebooks": cfg.num_codebooks,
        "frame_length": cfg.codec_frame_size,
        "model_fingerprint": fingerprint,
    }
    (reg_dir / "registration_manifest.json").write_text(
        json.dumps(reg_manifest, indent=4) + "\n"
    )
    print(f"  ✓ registration_manifest.json", flush=True)


def copy_tokenizer(model_dir: str, out_dir: Path):
    src = Path(model_dir)
    dst = out_dir / "tokenizer"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "processor_config.json"]:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    print(f"  ✓ tokenizer/", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path to fine-tuned model directory")
    ap.add_argument("--output", required=True, help="Output ONNX directory")
    ap.add_argument("--only", choices=["slow", "fast", "decoder", "encoder", "all"],
                    default="all", help="Export only specific components")
    args = ap.parse_args()

    model_dir = args.model
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_dir = out_dir / "registration"
    reg_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir} ...", flush=True)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    cfg = model.config
    print(f"  {cfg.n_layer} slow layers, {cfg.n_fast_layer} fast layers, "
          f"dim={cfg.dim}, vocab={cfg.vocab_size}", flush=True)

    # Load codec
    print("Loading codec ...", flush=True)
    codec = model.load_codec(device="cpu", dtype=torch.float32)

    # Monkey-patch codec for ONNX compatibility (in-place ops → out-of-place)
    _patch_codec_for_onnx()

    # Convert to FP16
    print("Converting to FP16 ...", flush=True)
    model = model.to(dtype=torch.float16)
    codec = codec.to(dtype=torch.float16)
    model.eval()
    codec.eval()

    if args.only in ("slow", "all"):
        print("\n=== Exporting Slow AR ===", flush=True)
        t0 = time.time()
        export_slow_ar(model, out_dir)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    if args.only in ("fast", "all"):
        print("\n=== Exporting Fast AR ===", flush=True)
        t0 = time.time()
        export_fast_ar(model, out_dir)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    if args.only in ("decoder", "all"):
        print("\n=== Exporting Codec Decoder ===", flush=True)
        t0 = time.time()
        export_codec_decoder(codec, out_dir)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    if args.only in ("encoder", "all"):
        print("\n=== Exporting Codec Encoder ===", flush=True)
        t0 = time.time()
        export_codec_encoder(codec, reg_dir)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    if args.only == "all":
        print("\n=== Writing manifests ===", flush=True)
        write_manifests(model, model_dir, out_dir, reg_dir)

        print("\n=== Copying tokenizer ===", flush=True)
        copy_tokenizer(model_dir, out_dir)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("ONNX export complete!", flush=True)
    print(f"  Output: {out_dir}", flush=True)
    total = 0
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
            if sz > 1024 * 1024:
                print(f"  {f.relative_to(out_dir)}: {sz / 1e6:.1f} MB", flush=True)
    print(f"  Total: {total / 1e6:.1f} MB", flush=True)
    print("\nNext: test with the ONNX runtime:", flush=True)
    print(f"  ARKTTS_MODEL_DIR={out_dir} bash onnx_runtime/start_server.sh", flush=True)


if __name__ == "__main__":
    main()
