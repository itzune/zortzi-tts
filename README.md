# zortzi-tts

Basque (`eu`) fine-tune of [Audio8_TTS](https://github.com/Audio8-AI/Audio8_TTS)
(the 0.6B DualAR TTS preview model). Two voices are exposed: **Maider** and
**Antton**. Ships with a Basque text normalizer / manifest builder, a
two-phase training recipe, and an ONNX export + INT4 quantization pipeline
that runs on CPU with the Audio8 ONNX runtime.

> **Voices.** The released model speaks **only** Basque with the **Maider** and
> **Antton** voices. These are anchored to the
> [HiTZ-Aholab](https://zenodo.org/records/17952596) speakers via reference
> clips; no other voices are registered.

## Why this exists

Audio8_TTS performs **no** language-specific text normalization. The only
upstream transform is whitespace collapsing in `clean_text()`. There is no
G2P, no number-to-words, no abbreviation expansion, no language-ID token.
Text is BPE-tokenized verbatim over a Qwen2-style ~151.6K vocab. So every
numeral / symbol / abbreviation that should be *spoken* must be normalized
**before** `audio8_tts_prepare.py` encodes the audio — i.e. in the manifest's
`text` field. That is this project's job.

See [`RESEARCH.md`](RESEARCH.md) for the full feasibility analysis.

---

## Table of contents

- [Install](#install)
- [Inference: GPU (PyTorch)](#inference-gpu-pytorch)
- [Inference: CPU (ONNX)](#inference-cpu-onnx)
- [Training: how zortzi-tts was trained](#training-how-zortzi-tts-was-trained)
- [ONNX export & INT4 quantization](#onnx-export--int4-quantization)
- [The text normalizer](#the-text-normalizer)
- [Manifest CLI](#manifest-cli)
- [Manifest schema](#manifest-schema)
- [Layout](#layout)
- [Known Issues](#known-issues)
- [Datasets, Licenses & Acknowledgements](#datasets-licenses--acknowledgements)

---

## Install

```bash
uv sync
```

## Inference: GPU (PyTorch)

For GPU inference, use the upstream PyTorch model directly. This is the
highest-fidelity path — full BF16, no quantization, KV-cache on device.

### Requirements

```bash
# On the GPU server (NVIDIA L40 / A100 / similar):
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend=cu124 \
    transformers soundfile torch
```

### Synthesize

```bash
.venv/bin/python /path/to/Audio8_TTS/audio8_tts_infer.py \
    --model /root/work/outputs/audio8_tts_sft_basque_final/ \
    --text "Kaixo mundua! Nire izena Maider da." \
    --reference-audio /root/work/data/hitz/maider/NEU_05850.wav \
    --reference-text "Aurrelaria prest dago jokatzeko." \
    --output out.wav \
    --temperature 0.8 --top-p 0.95
```

> **Sampling is required.** Do **not** pass `--greedy`. Greedy decoding causes
> repetition loops that never reach EOS. Use `--temperature 0.8 --top-p 0.95`.

### Voice reference clips

The voice is carried by the `--reference-audio` clip (its codec codes are the
primary voice carrier; the speaker token is always `<|speaker:0|>`). Use a
**prosodically neutral** clip to avoid bleeding acoustic style into the
output:

| Voice | Reference clip | Reference text |
|-------|----------------|----------------|
| Maider | `data/hitz/maider/NEU_05850.wav` | *Aurrelaria prest dago jokatzeko.* |
| Antton | `data/hitz/antton/NEU_11782.wav` | *Inguru hura gerrillarien esku izan da denbora luzean.* |

These were selected with `find_neutral_ref.py` (pitch-range/median ratio < 0.5)
so the reference doesn't contaminate prosody measurements or output style.

## Inference: CPU (ONNX)

For CPU deployment, use the exported ONNX models with the Audio8 ONNX runtime.
This runs entirely on CPU with no GPU or PyTorch dependency.

### Requirements

```bash
cd /path/to/Audio8_TTS/onnx_runtime
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python onnxruntime soundfile numpy scipy
```

### Synthesize

```bash
cd /path/to/Audio8_TTS/onnx_runtime
.venv/bin/python -m arktts_runtime.cli \
    --model-dir /root/work/outputs/onnx_p2_3k \
    --voices-dir /root/work/voices \
    --text "Kaixo mundua! Nire izena Maider da." \
    --voice maider \
    --output out.wav \
    --temperature 0.8 --top-p 0.95 \
    --max-new-tokens 512
```

Select the voice with `--voice maider` or `--voice antton`. These are the only
two registered voices.

### Precision: FP16 vs INT4

The ONNX export produces **FP16** models. An optional **INT4** weight
quantization step (see [below](#int4-quantization)) reduces the AR models
~2–3× and is actually **faster** on CPU (better cache utilization):

| Model | FP16 | INT4 | Notes |
|-------|------|------|-------|
| Slow AR | 1077 MB | 278 MB | 121 `MatMulNBits` + 11 `GatherBlockQuantized` (embeddings) |
| Fast AR | 134 MB | 34 MB | 21 `MatMulNBits` + 1 `GatherBlockQuantized` |
| Codec decoder | 266 MB | 266 MB | not quantized (Conv1d-based) |
| Codec encoder | 421 MB | 421 MB | not quantized (registration only) |
| **Inference total** | **1477 MB** | **575 MB** | matches official Audio8 INT4 (572 MB) |

Select precision at runtime:

```bash
# FP16 (default if only FP16 is available)
.venv/bin/python -m arktts_runtime.cli --model-dir .../onnx_p2_3k --precision fp16 ...

# INT4 (faster + smaller on CPU)
.venv/bin/python -m arktts_runtime.cli --model-dir .../onnx_p2_3k --precision int4 ...
```

The available precisions are declared in `runtime_manifest.json`
(`available_precisions`). ONNX Runtime upcasts FP16 → FP32 internally for
computation, so FP16 is a storage-only optimization on CPU.

> **Note:** INT4 quantization can degrade output quality (flatter prosody,
> less reliable question intonation) compared to FP16. Prefer FP16 when
> fidelity matters; use INT4 when size or CPU speed is the priority.

### CPU inference speed

On an 8-core CPU, a ~3s Basque utterance synthesizes in:

- FP16: ~23s
- INT4: ~17s (28% faster)

---

## Training: how zortzi-tts was trained

A **two-phase curriculum** on an NVIDIA L40 (48 GB). Full fine-tuning (not
LoRA) — new-language phonotactics require full weight updates.

### Phase 1 — phonotactics base (Common Voice 26.0)

**Goal:** learn the text→semantic (phonotactics) mapping for Basque, broadly,
across many speakers. No reference audio.

**Data:** Mozilla Common Voice 26.0 Basque, 134,531 validated clips (CC0).

**Critical: stop early.** CV is flat read-speech. Training too long entrenches
a neutral prosody prior (prosodic hysteresis) that Phase 2's conservative LR
cannot fully reverse. Phase 1 stops at the "Goldilocks" zone (~step 6000)
where phonotactics are learned but pretrained expressivity is still intact.

```bash
# 1. Download CV 26.0 eu, extract to /root/work/data/cv26/
# 2. Build manifest (text normalized upstream):
zortzi-manifest from-common-voice \
  --tsv /root/work/data/cv26/cv-corpus-26.0-2026-06-12/eu/validated.tsv \
  --clips /root/work/data/cv26/cv-corpus-26.0-2026-06-12/eu/clips \
  --output /root/work/data/cv26/train_cv26.jsonl \
  --min-upvotes 2 --max-duration-sec 20

# 3. Encode audio → codec codes (GPU):
python audio8_tts_prepare.py \
  --input-jsonl /root/work/data/cv26/train_cv26.jsonl \
  --output-jsonl /root/work/data/cv26/train_cv26_prepared.jsonl \
  --device cuda --dtype bfloat16 --batch-size 4

# 4. Train (stop at step 6000):
bash scripts/launch_cv_base.sh
```

**Phase 1 config:** LR 1e-5 cosine, batch 4×4, max_steps 6000, save every 1000.
~64 min runtime. Loss 49.2→34.5, slow_accuracy 10%→20%.

### Phase 2 — voice anchor + prosody polish (HiTZ, merged)

**Goal:** anchor Maider/Antton voice identity **and** restore expressive
prosody (especially wh-question intonation) in a single pass.

**Data:** HiTZ-Aholab Basque TTS (Maider + Antton, 27,000 clips, CC BY 4.0),
**upsampled** to 41,300 clips: declarative ×1, interrogative ×3, exclamative
×3. This is functionally equivalent to sequence-level loss weighting — it
biases training toward expressive prosody without re-encoding audio.

**References:** cross-clip self-conditioning (ref = a *different* clip from the
same speaker). This teaches the zero-shot voice-cloning path.

```bash
# 1. Build HiTZ manifest (both speakers) with self-references:
zortzi-manifest pair-references \
  --input /root/work/data/hitz/train_hitz.jsonl \
  --output /root/work/data/hitz/train_hitz_paired.jsonl \
  --speaker-col speaker

# 2. Encode audio → codec codes (GPU):
python audio8_tts_prepare.py \
  --input-jsonl /root/work/data/hitz/train_hitz_paired.jsonl \
  --output-jsonl /root/work/data/hitz/train_hitz_paired_prepared.jsonl \
  --device cuda --dtype bfloat16 --batch-size 4

# 3. Upsample expressive sentences (interrogatives/exclamatives ×3):
python scripts/upsample_expressive.py \
  --input  /root/work/data/hitz/train_hitz_paired_prepared.jsonl \
  --output /root/work/data/hitz/train_hitz_prosody_prepared.jsonl \
  --interrogative-factor 3 --exclamative-factor 3

# 4. Train from Phase 1 checkpoint:
bash scripts/launch_anchor.sh
```

**Phase 2 config:** LR 5e-6 linear decay, batch 4×4, 3 epochs, warmup 0.03,
save every 500. Started from Phase 1 step-6000 checkpoint.

### Training infrastructure

- **DeepSpeed ZeRO-2** — config in `scripts/deepspeed_zero2.json` (not shipped
  with the Audio8 repo; created manually).
- **No gradient checkpointing** — `ArkttsModel` does not support it. Disabled
  via `GRADIENT_CHECKPOINTING=false`.
- **Checkpoint audio probes** — `zortzi_callbacks.py` spawns
  `generate_samples.py` at every `on_save`, generating probe sentences and
  logging `wandb.Audio` + quantitative prosody metrics (f0_std_hz,
  focus_peak_hz, boundary_delta_hz) to the current run. Gated by
  `SAMPLE_CALLBACK=1`.
- **wandb** — entity `itzune`, project `zortzi-tts`.

### Key training decisions

| Decision | Rationale |
|----------|-----------|
| Full fine-tune over LoRA | New-language phonotactics require full weight updates |
| Two-phase curriculum | Phase 1 (CV) learns phonotactics; Phase 2 (HiTZ) anchors voice + prosody |
| Stop Phase 1 at step 6000 | Avoid prosodic hysteresis from flat CV read-speech |
| Upsample interrogatives/exclamatives 3× | Sequence-level loss weighting for prosody |
| Cross-clip self-conditioning | Teaches zero-shot voice-cloning path |
| Neutral reference clips for eval | Expressive refs bleed style into prosody measurements |
| Sampling (temp 0.8, top-p 0.95) | Greedy causes infinite repetition loops |

---

## ONNX export & INT4 quantization

The Audio8 repo ships the ONNX **runtime** (inference code) but **no export
script**. The pre-built ONNX model on HuggingFace uses the *original* (English)
weights. `scripts/export_onnx.py` exports *our* fine-tuned Basque model so it
runs on the same CPU runtime.

### Step 1: Export to ONNX (FP16)

Run on **CPU** (export on GPU risks OOM during training; the export itself is
not GPU-bound):

```bash
cd /path/to/Audio8_TTS
.venv/bin/python /root/work/zortzi-tts/scripts/export_onnx.py \
    --model /root/work/outputs/audio8_tts_sft_basque_final/ \
    --output /root/work/outputs/onnx_p2_3k
```

This produces:

```
onnx_p2_3k/
├── slow_ar_fp16.onnx(.data)            # Slow AR with KV cache (1.1 GB)
├── fast_ar_fp16.onnx(.data)            # Fast AR with KV cache (134 MB)
├── codec_decoder_fp16.onnx(.data)      # codes → audio (266 MB)
├── registration/
│   ├── codec_encoder_fp16.onnx(.data)  # audio → codes (421 MB, for voice registration)
│   └── registration_manifest.json
├── runtime_manifest.json               # precision & architecture metadata
└── tokenizer/
    └── tokenizer.json
```

**What the export script does** (the non-obvious parts):

- **Custom wrapper `nn.Module`s** (`SlowAROnnx`, `FastAROnnx`,
  `CodecDecoderOnnx`, `CodecEncoderOnnx`) match the exact ONNX runtime
  interface — manual attention, explicit KV-cache delta inputs/outputs, no
  Python control flow.
- **Manual attention bug** — the original implementation computed `Q·V^T`
  instead of `Q·K^T`. This was both an ONNX export issue (key_cache inputs
  appeared unused and were pruned) **and** a correctness bug. Fixed to
  `Q·K^T` in the wrappers.
- **Tracing with T=1 at position >0** — tracing with T>1 at positions 0..T-1
  makes `ScatterElements` overwrite all cache positions that attention reads,
  so the exporter sees cache inputs as unused and removes them. Tracing with
  T=1 at position 4 (with non-zero cache at positions 0–3) makes the cache
  read visible to the exporter.
- **7 monkey-patches** to the codec for dynamic shapes (`_patch_codec_for_onnx`):
  - `ArkttsCodecWindowTransformer` — out-of-place `&` (was in-place)
  - `_rope` — explicit cos/sin in float32 (was `aten::polar`, hardcoded bfloat16)
  - `ArkttsCodecLayerScale` — out-of-place mul (was in-place)
  - `ArkttsCausalConvTranspose1d` — negative indexing (was shape-dependent constant)
  - `ArkttsSnake1d` — inline ops (the jit.scripted `_arktts_snake` captured
    `x.shape` and baked in the traced length — direct cause of "10 by 70"
    broadcast errors)
  - `ArkttsCausalConv1d` — hardcoded `right=0` (`_extra_padding` used
    `x.shape[-1]` in Python arithmetic, baking in pad values; safe for stride=1)
  - `ArkttsResidualUnit` — no shape-dependent cropping

### Step 2: Register voices

Register Maider and Antton reference clips (encodes each clip to codec codes
via the ONNX codec encoder, saves to the `VoiceStore` format):

```bash
.venv/bin/python /root/work/zortzi-tts/scripts/register_voices.py
```

This creates `/root/work/voices/{maider,antton}/{codes.npy,meta.json}`.

### Step 3 (optional): INT4 quantization

Reduce AR model size ~2–3× and speed up CPU inference ~28%:

```bash
.venv/bin/python /root/work/zortzi-tts/scripts/quantize_int4.py \
    --input /root/work/outputs/onnx_p2_final/slow_ar_fp16.onnx \
    --output /root/work/outputs/onnx_p2_final/slow_ar_int4.onnx

.venv/bin/python /root/work/zortzi-tts/scripts/quantize_int4.py \
    --input /root/work/outputs/onnx_p2_final/fast_ar_fp16.onnx \
    --output /root/work/outputs/onnx_p2_final/fast_ar_int4.onnx
```

Then update `runtime_manifest.json` to advertise INT4:

```json
{
  "default_precision": "int4",
  "available_precisions": ["int4"],
  ...
}
```

**How INT4 quantization works.** Audio8's official INT4 models use standard
ONNX Runtime operators — `MatMulNBits` (com.microsoft domain, bits=4,
block_size=128) for linear weights and `GatherBlockQuantized` for embeddings.
These are **not** proprietary. The quantizer is
`onnxruntime.quantization.cuda_quantizer.CudaQuantizer` (works on CPU despite
the name).

The script runs two passes:
1. **Linear weights** — traces each `MatMul` → `Transpose` → weight
   initializer chain (skips attention Q×K^T and probs×V MatMuls, which have
   dynamic B from the KV cache). Quantizes each `[N, K]` weight with
   `matmulnbits_blockwise_quantize` → `qweight [N, K/128, 64]` (uint8),
   `scales [N, K/128]` (float16), `zero_points [N, ceil(K/128/2)]` (uint8,
   pre-packed 4-bit). Replaces `Transpose + MatMul` with a single `MatMulNBits`
   node. Produces 121 `MatMulNBits` nodes (slow AR) and 21 (fast AR).
2. **Embedding tables** — finds `Gather` nodes whose first input is a 2D float
   initializer with "embedding" in its name. Quantizes each `[vocab, dim]`
   table with `symmetric_blockwise_quantize` (block_size=32) → packed INT4
   `Q4` tensor, FP16 `scales`, INT4 `zero_points`. Replaces `Gather` with
   `GatherBlockQuantized` in the `com.microsoft` domain. Produces 11 nodes
   (slow AR) and 1 (fast AR).

This matches Audio8's official INT4 format exactly: total inference size
575 MB vs official 572 MB.

---

## The text normalizer

`basque_manifest/normalize.py` cleans text before it enters the manifest
(whitespace collapsing, punctuation, symbol expansion). It is applied when
building training manifests so transcripts are consistent.

```python
from basque_manifest import normalize_text

normalize_text("Kaixo  mundua!")
# -> "Kaixo mundua!"
```

Numeral-expansion rules exist but the model does not pronounce them reliably
in Basque — see [Known Issues](#known-issues).

## Manifest CLI

Build a raw manifest (the exact schema `audio8_tts_prepare.py` expects) from a
data source, with text normalized along the way:

```bash
# Common Voice 26.0 eu (validated split)
zortzi-manifest from-common-voice \
  --tsv /data/cv-corpus-26.0-eu/validated.tsv \
  --clips /data/cv-corpus-26.0-eu/clips \
  --output data/cv_eu.jsonl \
  --min-upvotes 2 --max-duration-sec 20

# Generic transcript set (covers OpenSLR SLR76, HiTZ Maider/Antton, anything
# with an id+text+audio mapping in a TSV/CSV)
zortzi-manifest from-transcripts \
  --transcripts /data/slr76/line_index.tsv \
  --audio-dir /data/slr76/wav \
  --audio-ext wav --id-col 0 --text-col 1 --audio-col 0 \
  --output data/slr76.jsonl

# Pair reference clips for zero-shot voice training (cross-clip self-conditioning)
zortzi-manifest pair-references \
  --input data/hitz.jsonl \
  --output data/hitz_paired.jsonl \
  --speaker-col speaker

# Merge sources into one training manifest
zortzi-manifest merge data/cv_eu.jsonl data/slr76.jsonl \
  --output data/train.jsonl

# (Re-)normalize an already-clean manifest's text in place
zortzi-manifest normalize data/train.jsonl --in-place

# Inspect
zortzi-manifest stats data/train.jsonl
```

Then feed the result to the upstream pipeline:

```bash
python /tmp/Audio8_TTS/audio8_tts_prepare.py \
  --input-jsonl data/train.jsonl \
  --output-jsonl prepared/train.jsonl \
  --device cuda --dtype bfloat16 --batch-size 4

TRAIN_JSONL=prepared/train.jsonl NPROC_PER_NODE=1 \
  bash /tmp/Audio8_TTS/audio8_tts_sft.sh
```

## Manifest schema (matches upstream `audio8_tts_prepare.py`)

Each line is one JSON object. Relative audio paths resolve from the manifest
directory; absolute paths are used by default.

```json
{"id":"utt_001","text":"Target transcript","audio":"/abs/path/target.wav","reference_audio":"/abs/path/ref.wav","reference_text":"Reference transcript"}
{"id":"utt_002","text":"Another transcript","audio":"/abs/path/another.wav"}
```

- `id` — filename-safe, used as the `.npy` codec basename (`[\w.-]+`).
- `audio` — required.
- `reference_audio` + `reference_text` — optional, must appear together
  (zero-shot voice-cloning path). For single-speaker studio corpora you can
  instead train without references.

## Layout

```
basque_manifest/
  normalize.py          Basque text normalization
  records.py            Record / Source dataclasses, shared I/O
  audio.py              optional ffprobe-based duration probing
  builder.py            normalize -> filter -> write JSONL
  sources/
    common_voice.py     Mozilla Common Voice eu TSV loader
    transcripts.py      generic TSV/CSV loader (OpenSLR, HiTZ, ...)
  cli.py                argparse subcommands
scripts/
  export_onnx.py        PyTorch → ONNX export (FP16, custom wrappers)
  quantize_int4.py      ONNX FP16 → INT4 (MatMulNBits + GatherBlockQuantized)
  register_voices.py    encode reference clips → VoiceStore (maider, antton)
  launch_cv_base.sh     Phase 1 training launcher (CV 26.0)
  launch_anchor.sh      Phase 2 training launcher (HiTZ upsampled)
  upsample_expressive.py  duplicate ?/! sentences N× for prosody
  find_neutral_ref.py   scan clips for prosodically neutral references
  generate_samples.py   checkpoint audio probe generator + prosody metrics
  zortzi_callbacks.py   HF Trainer callback: probe at every on_save → wandb
  analyze_wandb.py      pull training metrics from wandb API
  analyze_all_prosody.py  batch prosody analysis across probe checkpoints
  compare_prosody.py    compare prosody metrics between two probe sets
  deepspeed_zero2.json  DeepSpeed ZeRO-2 config (not in upstream repo)
  setup_env.sh          GPU server venv bootstrap
tests/
  test_normalize.py     normalization tests (65)
  test_builder.py       builder + merge + pair-references tests (14)
```

## Known Issues

- **Numerals are not pronounced in Basque.** Even with text normalization
  (digits expanded to Basque words), the model speaks numbers in a mix of
  languages rather than Basque. Spell out numbers manually if correct
  pronunciation is required.

## Datasets, Licences & Acknowledgements

`zortzi-tts` produces a fine-tuned derivative of the Audio8 TTS preview
model. The fine-tuned weights and all code in this repository are released
under **Apache-2.0**, inherited from the upstream model. The training data
carries its own (stricter) obligations, summarized below.

### Model

| Component | Role | License |
|---|---|---|
| [Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) | Base model (fine-tuned from) | Apache-2.0 |

The Audio8 TTS model uses a DualAR architecture inspired by
[Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech). Full attribution
and third-party notices are in the upstream
[LICENSE](https://github.com/Audio8-AI/Audio8_TTS/blob/main/LICENSE) and
[NOTICE](https://github.com/Audio8-AI/Audio8_TTS/blob/main/NOTICE) files.

### Training data

| Dataset | Role in training | License | Clips used |
|---|---|---|---|
| [Mozilla Common Voice 26.0](https://commonvoice.mozilla.org/) — Basque (`eu`) | Phase 1: phonotactics base run (no references) | **CC0 1.0** (Public Domain) | 134,531 train clips |
| [HiTZ-Aholab Basque TTS](https://zenodo.org/records/17952596) (Maider + Antton) | Phase 2: voice-anchor run (cross-clip self-references) | **CC BY 4.0** | 27,000 clips (both speakers) |
| [OpenSLR SLR76](https://www.openslr.org/76/) — Basque | *Superseded* by Common Voice for the base run | CC BY-SA 4.0 | (pilot only) |

**Redistribution note.** Because the CC BY 4.0 HiTZ-Aholab data conditions the
voice of the released model (Maider, Antton), any redistribution of the
fine-tuned weights or generated audio must carry the HiTZ attribution and
citation below. The CC0 Common Voice data imposes no such obligation, though
attribution is requested by the corpus authors.

### Citation

If you use `zortzi-tts` or build on its training recipe, please cite the base
model and the datasets:

```bibtex
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

@inproceedings{kjartansson-etal-2020-open,
    title     = "Open-Source High Quality Speech Datasets for {B}asque,
                 {C}atalan and {G}alician",
    author    = "Kjartansson, Oddur and Gutkin, Alexander and
                 Butryna, Alena and Demirsahin, Isin and Rivera, Clara",
    booktitle = "Proceedings of the 1st Joint Workshop on Spoken Language
                 Technologies for Under-resourced languages (SLTU) and
                 Collaboration and Computing for Under-Resourced Languages
                 (CCURL)",
    month     = may, year = "2020",
    address   = "Marseille, France",
    publisher = "European Language Resources Association",
    pages     = "21--27",
    url       = "https://aclanthology.org/2020.sltu-1.3/"
}
```

### Funding acknowledgements

The HiTZ-Aholab speech synthesis dataset was developed with funding from:

> The Ministerio para la Transformación Digital y de la Función Pública and
> Plan de Recuperación, Transformación y Resiliencia — Funded by EU —
> NextGenerationEU within the framework of the project ILENIA
> (ref. 2022/TL22/00215335), and by a grant from the Department of Culture
> and Language Policy of the Basque Government (IKER-GAITU project).

The `zortzi-tts` fine-tuning work itself is independent and not directly funded
by the above grants; they are acknowledged here solely because the voice-anchor
dataset carries the obligation.
