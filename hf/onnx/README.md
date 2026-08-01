---
language:
  - eu
license: apache-2.0
base_model:
  - itzune/zortzi-tts
  - Audio8/Audio8-TTS-Preview-0.6b
tags:
  - text-to-speech
  - tts
  - basque
  - euskara
  - audio
  - audio8
  - onnx
  - int4
  - quantized
pipeline_tag: text-to-speech
datasets:
  - mozilla-foundation/common_voice_17_0
  - language-and-voice-lab/hitz_aholab_eu_tts
---

# zortzi-tts-onnx

**ONNX export of [`itzune/zortzi-tts`](https://huggingface.co/itzune/zortzi-tts)** — the Basque fine-tune of Audio8-TTS-Preview-0.6b. Ships **INT4-quantized** models (575 MB total) that run entirely on CPU with the [Audio8 ONNX runtime](https://github.com/Audio8-AI/Audio8_TTS/tree/main/onnx_runtime). No GPU or PyTorch dependency.

Two voices: **Maider** (female) and **Antton** (male).

> 🏗️ **Source code, export & quantization scripts:** [github.com/itzune/zortzi-tts](https://github.com/itzune/zortzi-tts)

---

## Files

```
├── slow_ar_int4.onnx (+ .data)         # Slow AR, INT4 (278 MB)
├── fast_ar_int4.onnx (+ .data)         # Fast AR, INT4 (34 MB)
├── slow_ar_fp16.onnx (+ .data)         # Slow AR, FP16 (1077 MB) — optional, higher fidelity
├── fast_ar_fp16.onnx (+ .data)         # Fast AR, FP16 (134 MB) — optional, higher fidelity
├── codec_decoder_fp16.onnx             # codes → audio (266 MB)
├── registration/
│   └── codec_encoder_fp16.onnx         # audio → codes (voice registration only)
├── voices/
│   ├── maider/{codes.npy, meta.json}   # Maider reference voice codes
│   └── antton/{codes.npy, meta.json}   # Antton reference voice codes
├── runtime_manifest.json               # precision & architecture metadata
└── tokenizer/
    └── tokenizer.json
```

### Precision comparison

| Component | FP16 | INT4 | Notes |
|-----------|------|------|-------|
| Slow AR | 1077 MB | 278 MB | 121 `MatMulNBits` + 11 `GatherBlockQuantized` |
| Fast AR | 134 MB | 34 MB | 21 `MatMulNBits` + 1 `GatherBlockQuantized` |
| Codec decoder | 266 MB | 266 MB | not quantized (Conv1d-based) |
| **Inference total** | **1477 MB** | **575 MB** | INT4 matches official Audio8 (572 MB) |

> **Which to use?** INT4 is the default — 2.5× smaller and ~28% faster on CPU.
> FP16 gives slightly better prosody fidelity. If you need FP16, download the
> `*_fp16.onnx` files and set `default_precision: "fp16"` in
> `runtime_manifest.json`. **Prefer FP16 when fidelity matters; use INT4 when
> size or CPU speed is the priority.**

---

## Quick start (CPU — ONNX Runtime)

### Install

```bash
# Clone the Audio8_TTS repo for the ONNX runtime (inference code)
git clone https://github.com/Audio8-AI/Audio8_TTS.git
cd Audio8_TTS/onnx_runtime

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python onnxruntime soundfile numpy scipy
```

### Synthesize

```bash
# Download this model:
# huggingface-cli download itzune/zortzi-tts-onnx --local-dir ./zortzi-tts-onnx

.venv/bin/python -m arktts_runtime.cli \
    --model-dir ./zortzi-tts-onnx \
    --voices-dir ./zortzi-tts-onnx/voices \
    --text "Kaixo mundua! Nire izena Maider da." \
    --voice maider \
    --output out.wav \
    --temperature 0.8 --top-p 0.95 \
    --max-new-tokens 512
```

### Switch voices

```bash
# Antton
.venv/bin/python -m arktts_runtime.cli \
    --model-dir ./zortzi-tts-onnx \
    --voices-dir ./zortzi-tts-onnx/voices \
    --text "Kaixo mundua! Nire izena Antton da." \
    --voice antton \
    --output out.wav \
    --temperature 0.8 --top-p 0.95
```

### Select precision

```bash
# INT4 (default — smaller + faster)
.venv/bin/python -m arktts_runtime.cli --model-dir .../zortzi-tts-onnx --precision int4 ...

# FP16 (higher fidelity — requires the *_fp16.onnx files)
.venv/bin/python -m arktts_runtime.cli --model-dir .../zortzi-tts-onnx --precision fp16 ...
```

The available precisions are declared in `runtime_manifest.json`
(`available_precisions`). ONNX Runtime upcasts FP16 → FP32 internally for
computation, so FP16 is a storage-only optimization on CPU.

### Streaming

The Audio8 ONNX runtime includes a FastAPI streaming server:

```bash
cd /path/to/Audio8_TTS/onnx_runtime
.venv/bin/python -m arktts_runtime.service \
    --model-dir ./zortzi-tts-onnx \
    --voices-dir ./zortzi-tts-onnx/voices \
    --port 8024
```

- `POST /api/tts/stream` — streaming NDJSON with base64 PCM chunks
  (`chunk_frames=12`, ~557 ms of audio per chunk)
- `POST /v1/audio/speech` — OpenAI-compatible non-streaming
- `GET /api/health` — health check

### CPU inference speed

On an 8-core CPU, a ~3s Basque utterance synthesizes in:

| Precision | Time | RTF | Real-time? |
|-----------|------|-----|------------|
| INT4 | ~16s | 5.3× | ❌ no |
| FP16 | ~23s | 7.7× | ❌ no |

> CPU inference is **not real-time**. For real-time synthesis, use the
> [PyTorch model](https://huggingface.co/itzune/zortzi-tts) on GPU (RTF ≈ 1.05×
> on an NVIDIA L40). CPU ONNX is intended for offline/batch generation where
> latency is acceptable.

---

## How INT4 quantization works

Audio8's official INT4 models use standard ONNX Runtime operators —
`MatMulNBits` (com.microsoft domain, bits=4, block_size=128) for linear weights
and `GatherBlockQuantized` for embeddings. These are **not** proprietary. The
quantization script ([`quantize_int4.py`](https://github.com/itzune/zortzi-tts))
runs two passes:

1. **Linear weights** — `MatMul` → `MatMulNBits` (121 nodes slow AR, 21 fast AR)
2. **Embedding tables** — `Gather` → `GatherBlockQuantized` (11 nodes slow AR,
   1 fast AR)

This matches Audio8's official INT4 format exactly: 575 MB vs 572 MB official.

---

## Limitations & known issues

- **Numerals are not pronounced in Basque.** Even with text normalization, the
  model pronounces numbers in a mix of languages. Spell out numbers manually.
- **No built-in text normalization.** Expand acronyms (e.g. "TTS" → "te te ese")
  upstream of the model.
- **INT4 can degrade quality.** Flatter prosody and less reliable question
  intonation compared to FP16. Prefer FP16 when fidelity matters.
- **Two voices only.** Maider and Antton.
- **Sampling required.** Always use `--temperature 0.8 --top-p 0.95`. Do not
  use greedy decoding (repetition loops).
- **Not real-time on CPU.** Use GPU PyTorch for real-time synthesis.

---

## Model details

| | |
|---|---|
| **Architecture** | DualAR TTS (Qwen2-style decoder + codec) |
| **Parameters** | 601M (AR) + codec |
| **Opset** | 17 |
| **Codebooks** | 10 × 4096 entries |
| **Audio** | 44.1 kHz, ~21.5 frames/s |
| **Max sequence length** | 2048 |
| **PyTorch source** | [`itzune/zortzi-tts`](https://huggingface.co/itzune/zortzi-tts) |
| **Base model** | [`Audio8/Audio8-TTS-Preview-0.6b`](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) (Apache-2.0) |
| **License** | Apache-2.0 (inherited from base) |

---

## Datasets, licenses & acknowledgements

| Dataset | Role | License | Clips |
|---|---|---|---|
| [Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) | Base model | Apache-2.0 | — |
| [Mozilla Common Voice 26.0](https://commonvoice.mozilla.org/) — Basque ¹ | Phase 1: phonotactics | **CC0 1.0** | 134,531 |
| [HiTZ-Aholab Basque TTS](https://zenodo.org/records/17952596) (Maider + Antton) | Phase 2: voice + prosody | **CC BY 4.0** | 27,000 |

> ¹ Common Voice 26.0 was obtained directly from the Common Voice website.
> The closest HuggingFace dataset mirror is
> [`mozilla-foundation/common_voice_17_0`](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0).

> **Redistribution note.** Because the CC BY 4.0 HiTZ-Aholab data conditions
> the voice of this model (Maider, Antton), any redistribution of the
> fine-tuned weights or generated audio must carry the HiTZ attribution and
> citation below.

### Funding acknowledgements

The HiTZ-Aholab speech synthesis dataset was developed with funding from:

> The Ministerio para la Transformación Digital y de la Función Pública and
> Plan de Recuperación, Transformación y Resiliencia — Funded by EU —
> NextGenerationEU within the framework of the project ILENIA
> (ref. 2022/TL22/00215335), and by a grant from the Department of Culture
> and Language Policy of the Basque Government (IKER-GAITU project).

## Citation

```bibtex
@dataset{itzune_zortzi_tts_onnx,
    author    = {{itzune}},
    title     = {{zortzi-tts-onnx: ONNX INT4 export of zortzi-tts (Basque TTS)}},
    year      = 2025,
    publisher = {Hugging Face},
    url       = {https://huggingface.co/itzune/zortzi-tts-onnx}
}

@inproceedings{ardila-etal-2020-common,
    title     = "Common Voice: A Massively-Multilingual Speech Corpus",
    author    = "Ardila, Rosana and Branson, Megan and Davis, Kelly and
                 Kohler, Michael and Meyer, Josh and Henretty, Michael and
                 Morais, Reuben and Saunders, Lindsay and Tyers, Francis and
                 Weber, Gregor",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation
                 Conference (LREC)",
    month     = may, year = "2020",
    address   = "Marseille, France",
    publisher = "European Language Resources Association",
    pages     = "4218--4222",
    url       = "https://aclanthology.org/2020.lrec-1.520/"
}

@dataset{navas_hernaez_2025_17952596,
    author    = {Navas, Eva and Hernaez Rioja, Inmaculada and
                 Saratxaga, Ibon and Sanchez, Jon and
                 García Romillo, Víctor and Flores Ríos, Mariana and
                 Bellanco, Aitor},
    title     = {{HiTZ-Aholab speech synthesis dataset in Basque}},
    month     = dec, year      = 2025,
    publisher = {Zenodo},
    version   = {1.0},
    doi       = {10.5281/zenodo.17952596},
    url       = {https://doi.org/10.5281/zenodo.17952596}
}
```
