---
language:
  - eu
license: apache-2.0
base_model: Audio8/Audio8-TTS-Preview-0.6b
tags:
  - text-to-speech
  - tts
  - basque
  - euskara
  - audio
  - audio8
  - fine-tuned
library_name: transformers
pipeline_tag: text-to-speech
datasets:
  - mozilla-foundation/common_voice_17_0
  - language-and-voice-lab/hitz_aholab_eu_tts
---

# zortzi-tts

**Basque (`eu`) fine-tune of [Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)** — a 0.6B DualAR text-to-speech model. This checkpoint speaks **only Basque** with two voices: **Maider** (female) and **Antton** (male), anchored to the [HiTZ-Aholab](https://zenodo.org/records/17952596) speakers.

> 🏗️ **Source code & training recipe:** [github.com/itzune/zortzi-tts](https://github.com/itzune/zortzi-tts)

---

## Model details

| | |
|---|---|
| **Architecture** | DualAR TTS (Qwen2-style decoder + codec) |
| **Parameters** | 601M (AR) + codec |
| **Dtype** | BF16 |
| **Vocab size** | 155,776 (BPE, Qwen2-style) |
| **Codebooks** | 10 × 4096 entries |
| **Audio** | 44.1 kHz, ~21.5 frames/s |
| **Max sequence length** | 2048 (~95s hard ceiling; keep clips ≤20s for stable training) |
| **Base model** | [`Audio8/Audio8-TTS-Preview-0.6b`](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) (Apache-2.0) |
| **License** | Apache-2.0 (inherited from base) |

### Voices

Only **two voices** are available. They are carried by reference audio clips
(their codec codes are the primary voice carrier; the speaker token is always
`<|speaker:0|>`).

| Voice | Gender | Reference clip | Reference text |
|-------|--------|----------------|----------------|
| **Maider** | Female | `voices/maider.wav` | *Aurrelaria prest dago jokatzeko.* |
| **Antton** | Male | `voices/antton.wav` | *Inguru hura gerrillarien esku izan da denbora luzean.* |

These reference clips are included in this repository under `voices/`. They
were selected with `find_neutral_ref.py` (pitch-range/median ratio < 0.5) so
that the reference does not bleed expressive style into the output.

---

## Quick start (GPU — PyTorch)

This is the highest-fidelity path: full BF16, no quantization, KV-cache on device.

### Install

```bash
# Clone the Audio8_TTS runtime (provides audio8_tts_infer.py + custom modeling code)
git clone https://github.com/Audio8-AI/Audio8_TTS.git
cd Audio8_TTS

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend=cu124 \
    transformers soundfile torch
```

### Synthesize

```bash
.venv/bin/python audio8_tts_infer.py \
    --model itzune/zortzi-tts \
    --text "Kaixo mundua! Nire izena Maider da." \
    --reference-audio voices/maider.wav \
    --reference-text "Aurrelaria prest dago jokatzeko." \
    --output out.wav \
    --temperature 0.8 --top-p 0.95
```

> ⚠️ **Sampling is required.** Do **not** pass `--greedy`. Greedy decoding
> causes repetition loops that never reach EOS. Always use
> `--temperature 0.8 --top-p 0.95`.

### Switch voices

Change the `--reference-audio` and `--reference-text` to the Antton clip:

```bash
.venv/bin/python audio8_tts_infer.py \
    --model itzune/zortzi-tts \
    --text "Kaixo mundua! Nire izena Antton da." \
    --reference-audio voices/antton.wav \
    --reference-text "Inguru hura gerrillarien esku izan da denbora luzean." \
    --output out.wav \
    --temperature 0.8 --top-p 0.95
```

### Programmatic usage

```python
import torch
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    "itzune/zortzi-tts",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).cuda()
processor = AutoProcessor.from_pretrained("itzune/zortzi-tts", trust_remote_code=True)
```

See the upstream [`audio8_tts_infer.py`](https://github.com/Audio8-AI/Audio8_TTS/blob/main/audio8_tts_infer.py) for the full generation loop.

---

## ONNX deployment (CPU)

For CPU deployment without PyTorch/GPU, see the companion model card:
**[`itzune/zortzi-tts-onnx`](https://huggingface.co/itzune/zortzi-tts-onnx)** —
INT4-quantized ONNX models (575 MB total) that run on the Audio8 ONNX runtime.

---

## Training summary

A **two-phase curriculum** with full fine-tuning (not LoRA) on an NVIDIA L40
(48 GB). Full details in the [source repo](https://github.com/itzune/zortzi-tts).

### Phase 1 — phonotactics base (Common Voice 26.0)

Learned the text→semantic (phonotactics) mapping for Basque across many
speakers, with **no reference audio**.

- **Data:** Mozilla Common Voice 26.0 Basque, 134,531 validated clips (CC0)
  *(v26.0 was downloaded directly from the [Common Voice website](https://commonvoice.mozilla.org/); the closest HuggingFace mirror is [`common_voice_17_0`](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0))*
- **Config:** LR 1e-5 cosine, batch 4×4, max_steps 6000
- **Runtime:** ~64 min
- **Key insight:** stopped at step 6000 (the "Goldilocks" zone) to avoid
  prosodic hysteresis — CV is flat read-speech, and training too long entrenches
  a neutral prosody prior that later phases cannot fully reverse

### Phase 2 — voice anchor + prosody polish (HiTZ, merged)

Anchored Maider/Antton voice identity **and** restored expressive prosody
(especially wh-question intonation) in a single pass.

- **Data:** HiTZ-Aholab Basque TTS (Maider + Antton, 27,000 clips, CC BY 4.0),
  **upsampled** to 41,300 clips: declarative ×1, interrogative ×3, exclamative ×3
  (functionally equivalent to sequence-level loss weighting)
- **References:** cross-clip self-conditioning (ref = a *different* clip from
  the same speaker — teaches the zero-shot voice-cloning path)
- **Config:** LR 5e-6 linear decay, batch 4×4, 3 epochs, warmup 0.03
- **Started from:** Phase 1 step-6000 checkpoint
- **Final:** 7,746 steps, train_loss=29.96, slow_accuracy=30.9%

### Key decisions

| Decision | Rationale |
|----------|-----------|
| Full fine-tune over LoRA | New-language phonotactics require full weight updates |
| Two-phase curriculum | Phase 1 (CV) learns phonotactics; Phase 2 (HiTZ) anchors voice + prosody |
| Stop Phase 1 at step 6000 | Avoid prosodic hysteresis from flat CV read-speech |
| Upsample interrogatives/exclamatives 3× | Sequence-level loss weighting for prosody |
| Cross-clip self-conditioning | Teaches zero-shot voice-cloning path |
| Sampling (temp 0.8, top-p 0.95) | Greedy causes infinite repetition loops |

---

## Limitations & known issues

- **Numerals are not pronounced in Basque.** Even with text normalization
  (digits expanded to Basque words), the model pronounces numbers in a mix of
  languages rather than Basque. Spell out numbers manually if correct
  pronunciation is required.
- **No built-in text normalization at inference.** Audio8_TTS performs no
  language-specific G2P, number-to-words, or abbreviation expansion. Text is
  BPE-tokenized verbatim. Acronyms like "TTS" should be expanded to "te te ese"
  upstream of the model.
- **Two voices only.** This checkpoint speaks only Basque with Maider and
  Antton. No other voices are registered.
- **Sampling required.** Greedy decoding produces repetition loops.

---

## Datasets, licenses & acknowledgements

This model is a fine-tuned derivative of the Audio8 TTS preview model. The
fine-tuned weights are released under **Apache-2.0**, inherited from the
upstream model. The training data carries its own obligations:

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

The `zortzi-tts` fine-tuning work itself is independent and not directly funded
by the above grants; they are acknowledged here solely because the voice-anchor
dataset carries the obligation.

## Citation

If you use this model or build on its training recipe, please cite the base
model, the datasets, and this work:

```bibtex
@dataset{itzune_zortzi_tts,
    author    = {{itzune}},
    title     = {{zortzi-tts: Basque fine-tune of Audio8-TTS-Preview-0.6b}},
    year      = 2025,
    publisher = {Hugging Face},
    url       = {https://huggingface.co/itzune/zortzi-tts}
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
