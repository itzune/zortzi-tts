# Fine-tuning Audio8_TTS for Basque on an NVIDIA L40 — Research Notes

Date compiled: 2026-07-31
Scope: technical viability of adding Basque (eu) support to Audio8_TTS via
supervised fine-tuning, plus candidate datasets.

---

## 1. What Audio8_TTS is

- **Repo**: [Audio8-AI/Audio8_TTS](https://github.com/Audio8-AI/Audio8_TTS) —
  "SOTA-Class TTS at Compact Scale", Apache-2.0 license.
- **Checkpoint**: 0.6B preview weights, mirrored under two HF orgs —
  [AutoArk-AI/Audio8-TTS-Preview-0.6b](https://huggingface.co/AutoArk-AI/Audio8-TTS-Preview-0.6b)
  and [Audio8/Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b).
  Related/rebranded upstream repo: [AutoArk/audio8-tts-0.6B](https://github.com/AutoArk/audio8-tts-0.6B).
- **Architecture**: DualAR (Fish Audio S2 Pro–style)
  - Main model: 601,159,424 params (excl. codec)
  - Slow AR: 24 layers, width 896, 14 attention heads, 2 KV heads — predicts one semantic token/audio frame
  - Fast AR: 4 layers, width 896, 14 heads, 2 KV heads — predicts the frame's 10 codec codebooks conditioned on the slow hidden state
  - Codec: 44.1 kHz, 2,048 samples/frame (~21.5 frames/s), 10 codebooks × 4,096 entries
  - Context: up to 2,048 packed text/audio positions
  - Weights stored in BF16 (~1.2 GB `model.safetensors`), plus a separate `codec.pth` (~1.35 GB)
- **Supported languages (Preview release)**: Cantonese, Chinese, Dutch,
  English, French, German, Italian, Japanese, Korean, Polish, Spanish.
  **Basque is not included** — this would be new-language adaptation, not
  fine-tuning an existing voice.
- **Benchmarks**: Best-in-class EN WER and competitive ZH CER on Seed-TTS
  versus Fish S2 Pro (4.6B), Higgs Audio v2 (4.7B), CosyVoice3 (1.5B),
  MOSS-TTS (8.5B), VoxCPM2 (2.3B) — despite being the smallest model in the
  comparison.

### 1.1 Text pipeline (verified from source, not assumed)

I pulled the actual processor code,
[`processing_arktts.py`](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b/blob/main/processing_arktts.py),
from the HF repo. Key findings:

- **No phonemizer / G2P step anywhere in the pipeline.** Text is only
  whitespace-cleaned (`_clean_text`) and passed straight into
  `AutoTokenizer.encode(...)`, a standard fast BPE tokenizer
  (`tokenizer.json` is 12.2 MB). `semantic_begin_id = 151678` strongly
  suggests a Qwen2/2.5-style ~151.6K-token vocabulary with audio/semantic
  tokens appended after the text vocab.
- Prompts are built with a chat-style template:
  `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|voice|>`,
  optionally prefixed with a reference-text/reference-audio block and a
  `<|speaker:0|>` tag for voice cloning.
- **No explicit language-ID token** was found in this file — language
  identity is presumably learned implicitly from text+audio pairing during
  training, not signaled by a discrete tag.
- **Practical implication for Basque**: Since it's raw BPE over UTF-8 text
  with no phoneme layer, Basque's orthography (plain Latin alphabet, digraphs
  such as `tt`, `dd`, `tx`, `ts`, `tz`; agglutinative morphology; no
  diacritics beyond rare borrowings) will tokenize mechanically without
  errors. But the model has **zero learned grapheme→pronunciation mapping**
  for Basque — all of that must come from fine-tuning data. This confirms
  the risk flagged earlier: this is closer to teaching a new language from
  scratch (within the model's existing capacity) than nudging an existing
  voice.

### 1.2 Real-world fine-tuning report found (independent, not from Anthropic)

A third-party benchmark/blog post,
["Audio8 TTS 0.6B on macOS and RTX 3090 Ti"](https://instavar.com/blog/ai-production-stack/Audio8_TTS_0_6B_MacOS_RTX_3090_Ti_Benchmark_2026),
ran inference and a small LoRA fine-tuning pilot on this exact checkpoint:

- Inference on an RTX 3090 Ti peaked at **~2,938 MB** GPU memory for a
  2-item batch — consistent with the model's small size.
- They ran a **128-clip LoRA pilot** adapting toward a custom voice profile.
  Validation loss improved, but they observed **unstable end-of-speech
  (EOS) behavior** on longer training runs — i.e., a saved checkpoint with
  good loss numbers wasn't necessarily a usable voice; they had to
  listening-test outputs, not just trust the loss curve.
- Takeaway for a Basque fine-tune: budget time for **generation quality
  checks throughout training**, not just loss/WER tracking, and watch
  specifically for EOS/termination instability, which seems to be a known
  soft spot of this checkpoint under fine-tuning.

This is a single external, non-Anthropic source and should be treated as
anecdotal evidence rather than a guarantee — but it's a directly relevant
data point since it fine-tuned the identical model.

---

## 2. GPU feasibility — NVIDIA L40 (48 GB GDDR6, Ada Lovelace, ~90 TFLOPS BF16)

**Verdict: comfortably feasible**, and likely with a large safety margin,
given the model's small size and the independent report above showing full
inference fits in ~3 GB.

Rough training memory budget (mixed-precision AdamW, full fine-tune):

| Component | Approx. size |
|---|---|
| BF16 model weights | ~1.2 GB |
| FP32 master weights + gradients | ~4.8 GB |
| Adam optimizer states (m, v, FP32) | ~4.8 GB |
| Activations (grad checkpointing, 2048-token ctx) | a few GB, scales with batch size |
| **Total** | **~12–20 GB**, well inside 48 GB |

Notes:

- The repo's own training script (`audio8_tts_sft.sh`) already supports a
  single-GPU path (`NPROC_PER_NODE=1`), and multi-GPU only via
  `NPROC_PER_NODE=8` + gradient accumulation — an L40 sits well above the
  minimum bar implied by that single-GPU default.
- `FREEZE_SLOW_AR=true` / `FREEZE_FAST_AR=true` env flags let you fine-tune
  only one branch, useful if you want to reduce compute or isolate which
  branch is responsible for pronunciation vs. acoustic detail.
- No LoRA path ships in the official repo, but the third-party benchmark
  above patched in a LoRA path for their pilot — worth checking their post
  if you want to avoid full fine-tuning.
- Given the margin available on a 48 GB card, prefer **full fine-tuning**
  over LoRA for a new-language adaptation (more capacity to learn genuinely
  new phonotactics), reserving LoRA only if forgetting of the other 11
  languages becomes a hard requirement to avoid.

---

## 3. Basque datasets

| Dataset | Size | Type / quality | License | Link |
|---|---|---|---|---|
| **Common Voice 25.0 (eu)** | 702.03 h total, 472.44 h validated, 11,048 speakers, 464,733 clips | Crowdsourced, variable mic quality/noise, multi-speaker | CC0 | [Mozilla Data Collective](https://datacollective.mozillafoundation.org/datasets/cmn2hwe0d01n8mm07wug9r5he) |
| **OpenSLR SLR76** | Female set (1.6 GB) + male set (1.4 GB) compressed audio | Crowdsourced but higher/consistent quality than raw CV; multi-speaker | CC BY-SA 4.0 | [openslr.org/76](https://openslr.org/76/) |
| **HiTZ/Aholab neutral TTS corpora (Maider, Antton voices)** | Studio-quality, 2 professional native speakers | Clean single-speaker; already used to train existing VITS-based [aHoTTS](https://github.com/hitz-zentroa/aHoTTS) Basque voices | CC BY 4.0 (per HiTZ HF model cards) | [HiTZ/TTS-eu_antton](https://huggingface.co/HiTZ/TTS-eu_antton) |
| **HiTZ/Aholab emotional speech corpus** | 2 speakers (Maider, Antton), 4 emotion categories, studio quality | Expressive/emotional read speech, orthographic transcripts, Basque-specific spelling phenomena (e.g. "tt") covered | Check Zenodo record for exact terms | [Zenodo record 18804769](https://zenodo.org/records/18804769) |
| **Aholab historical Emotional Speech Database** (Saratxaga et al. 2006) | ~20 hours | Older but high-quality studio emotional recordings | Academic — verify availability/terms directly with Aholab | Referenced in [ACL Anthology 2020.sltu-1.3](https://aclanthology.org/2020.sltu-1.3.pdf) |
| **Basque Parliament corpus** (part of the 548h ASR training set) | Large, exact TTS-usable hours unclear | Spontaneous political speech, multi-speaker, non-neutral prosody — good for vocabulary/phonetic coverage, weak fit for clean TTS targets | Unclear — verify before use | Mentioned in [HiTZ Zentroa ASR announcement](https://www.hitz.eus/en/node/342) |
| **AhoTTS system + docs** (existing Basque TTS system for reference) | n/a (system, not corpus) | Useful as a quality/pronunciation reference baseline, and its GitHub lists corpus provenance for its bundled voices | Apache/CC mix, see repo | [hitz-zentroa/aHoTTS](https://github.com/hitz-zentroa/aHoTTS), [HiTZ TTS page](https://www.hitz.eus/en/tts) |

### Confirmed by HiTZ (Basque Center for Language Technology, Aholab Lab, UPV/EHU)

HiTZ's own Basque ASR system was trained on <cite index="9-1">548 hours of Basque voices from different public sources (Mozilla Common Voice 16.1, Basque Parliament, OpenSLR), reaching WER below 5%</cite> — a useful sanity check that this combination of sources is workable at scale for Basque speech modeling, even though ASR and TTS have different data-quality requirements.

The HiTZ emotional TTS corpus is explicitly described as containing <cite index="4-1">studio-quality audio recordings and orthographic transcriptions of read speech produced by professional native Basque speakers, recorded in a professional studio under controlled acoustic conditions</cite>, with <cite index="4-1">balanced coverage of multiple basic emotions</cite> and explicit handling of <cite index="4-1">Basque-specific spelling phenomena such as "tt"</cite> — directly relevant given Audio8_TTS's raw-text (no phonemizer) pipeline.

### Suggested recipe

1. **Bulk phonetic/prosodic coverage**: filtered Common Voice (validated
   split, then further filtered for SNR/clip length) + OpenSLR SLR76, for
   breadth of speakers and vocabulary.
2. **Quality anchor / final pass**: HiTZ Aholab Maider/Antton studio corpora
   (neutral + emotional) for clean single-speaker fine-tuning, closer to how
   the base checkpoint's own strong Seed-TTS numbers were likely achieved
   (clean few-speaker studio data).
3. **Optional domain coverage**: Basque Parliament corpus only if you need
   broader lexical/topic coverage and can tolerate noisier prosody — verify
   its license before use.
4. **Mitigate forgetting**: mix in a small proportion of one or two of the
   11 already-supported languages (Spanish/French are geographically and
   phonetically closest to Basque-speaking regions) during fine-tuning, and
   re-evaluate the other languages' WER/SIM after training to check for
   regression.
5. **Verify licenses before any redistribution** — CC0 (Common Voice),
   CC BY-SA 4.0 (OpenSLR), CC BY 4.0 (HiTZ) all permit research/fine-tuning
   use but differ on redistribution/share-alike terms.

---

## 4. Open questions / things not yet verified

- Exact hour count of usable single-speaker material in OpenSLR SLR76 (page
  gives file sizes, not hours).
- Whether the Basque Parliament corpus is legally clear for model training
  (not just ASR benchmarking) — contact HiTZ/Aholab to confirm.
- ~~Whether `audio8_tts_prepare.py`/`audio8_tts_sft.py` contain any
  language-specific text normalization~~ — **RESOLVED (2026-07-31):** the
  repo was cloned (`git clone https://github.com/Audio8-AI/Audio8_TTS.git`)
  and all four training scripts read end-to-end. The *only* upstream text
  transform is `clean_text()` (whitespace collapse). There is **no**
  number-to-words, abbreviation expansion, G2P, case folding, or language
  tag anywhere in `audio8_tts_data.py`, `audio8_tts_prepare.py`,
  `audio8_tts_sft.py`, or `audio8_tts_sft.sh` (confirmed by reading + a
  repo-wide grep for normaliz/abbrev/phonem/g2p/num2word). Every `num_*` /
  `number` hit counts codebooks or layers, not numerals. All Basque
  numeral/symbol/abbreviation handling must therefore happen in the
  manifest `text` field *before* `prepare` encodes audio. See §6–§7.
- Native-speaker audit of the Basque number normalizer (§7): the lexical
  forms of 17–19 (`hamazazpi` / `hemezortzi` / `hemeretzi` vs dialectal
  variants — Omniglot lists `hamazortzi` for 18, standard batua is
  `hemezortzi`), the multi-group "eta" placement for numbers like 1122,
  and morphological suffix adjustment for digit-adjacent Basque case
  endings (e.g. `%20ean` → `hogeiean` under the current verbatim-suffix
  heuristic).
- Confirm the exact on-disk layout of OpenSLR SLR76 and the HiTZ Aholab
  Maider/Antton releases (transcript filename, audio directory, column
  order) against the actual downloads, so the generic `from-transcripts`
  loader columns can be pinned.
- The third-party LoRA/EOS-instability report is a single external source,
  not independently reproduced here — treat as a signal to watch for, not a
  confirmed defect.

---

## 5. Reference links (all sources used)

- Audio8_TTS GitHub: https://github.com/Audio8-AI/Audio8_TTS
- Audio8_TTS (rebranded/upstream) GitHub: https://github.com/AutoArk/audio8-tts-0.6B
- HF checkpoint (AutoArk-AI): https://huggingface.co/AutoArk-AI/Audio8-TTS-Preview-0.6b
- HF checkpoint (Audio8 org, includes verified processor source): https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b
- `processing_arktts.py` source (fetched and verified): https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b/blob/main/processing_arktts.py
- Third-party fine-tuning/inference benchmark: https://instavar.com/blog/ai-production-stack/Audio8_TTS_0_6B_MacOS_RTX_3090_Ti_Benchmark_2026
- Common Voice 25.0 Basque datasheet: https://datacollective.mozillafoundation.org/datasets/cmn2hwe0d01n8mm07wug9r5he
- OpenSLR SLR76 (Basque): https://openslr.org/76/
- HiTZ TTS-eu_antton model card: https://huggingface.co/HiTZ/TTS-eu_antton
- HiTZ Aholab emotional Basque speech corpus (Zenodo): https://zenodo.org/records/18804769
- HiTZ aHoTTS GitHub: https://github.com/hitz-zentroa/aHoTTS
- HiTZ Zentroa Speech/Audio processing page: https://www.hitz.eus/en/tts
- HiTZ Basque ASR announcement (548h training data breakdown): https://www.hitz.eus/en/node/342
- HiTZ Whisper Large-v2 Basque model card: https://huggingface.co/HiTZ/whisper-large-v2-eu
- HiTZ Whisper Large-v3 Basque model card: https://huggingface.co/HiTZ/whisper-large-v3-eu
- Open-source Basque speech datasets survey (ACL Anthology): https://aclanthology.org/2020.sltu-1.3.pdf
- coqui-ai open-speech-corpora list: https://github.com/coqui-ai/open-speech-corpora
- Basque numbers (vigesimal tables, cross-checked for the normalizer): https://omniglot.com/language/numbers/basque.htm

---

## 6. Training pipeline — verified from the GitHub repo

The repo was cloned to `/tmp/Audio8_TTS` and all four training scripts read
end-to-end: `audio8_tts_sft.sh`, `audio8_tts_prepare.py`, `audio8_tts_data.py`,
`audio8_tts_sft.py`. This resolves the §4 normalization question and pins down
the exact manifest schema and training mechanics.

### 6.1 Three-stage pipeline

```
raw JSONL  →  audio8_tts_prepare.py  →  prepared JSONL + .npy codes  →  audio8_tts_sft.sh / sft.py
```

### 6.2 Raw manifest schema (the input we must produce)

```json
{"id":"utt_001","text":"Target transcript","audio":"audio/target.wav","reference_audio":"audio/ref.wav","reference_text":"Reference transcript"}
{"id":"utt_002","text":"Another transcript","audio":"audio/another.wav"}
```

- `id`: must match `[\w.-]+` (Unicode-safe) and be filename-safe — used as
  the `.npy` codec basename. Duplicates are rejected.
- `audio`: required. `reference_audio` + `reference_text`: optional but
  **must appear together** (zero-shot voice-cloning path). For
  single-speaker studio corpora you can instead train with no reference.
- Relative audio paths resolve from the manifest directory; absolute paths
  are accepted too. Audio is auto-resampled to the codec's 44.1 kHz, made
  mono via channel-mean, and NaN/inf samples are rejected.

### 6.3 Text normalization (the key finding)

The *only* upstream transform is:

```python
def clean_text(value, *, field_name="text") -> str:
    text = " ".join(str(value).strip().split())   # whitespace-collapse only
    if not text: raise ValueError(...)
    return text
```

No number-to-words, no abbreviation expansion, no G2P, no case folding, no
language tag, no per-language branch — anywhere. So **all** Basque
numeral/currency/symbol/abbreviation handling is the dataset builder's
responsibility, baked into the manifest `text` field before `prepare` runs.
This makes the HiTZ Aholab studio corpora (already spoken-word transcripts)
the lowest-risk `text` source, and Common Voice (raw digits/symbols in crowd
transcripts) the one most in need of normalization.

### 6.4 Prepare step (`audio8_tts_prepare.py`)

Loads the codec via `AutoModel(..., trust_remote_code=True).load_codec(...)`,
encodes each clip to `[10, T]` int32 codec indices, saved atomically as
`{id}.target.npy` (and `{id}.reference.npy` for voice cloning). Idempotent —
reuses valid existing arrays unless `--overwrite`. Batch encode failures fall
back to per-item encoding and are logged to `failures.jsonl` (not fatal unless
*all* samples fail). Run on the L40 with `--device cuda --dtype bfloat16`.

### 6.5 Training (`audio8_tts_sft.py`, HF Trainer + `torch.distributed.run`)

- **DualAR loss** (`Audio8TTSTrainer.compute_loss`):
  `slow_loss_weight·slow_CE + fast_loss_weight·fast_CE` (both default 1.0).
  Slow CE is over semantic/EOS tokens (row 0); fast CE is the 10 codebooks
  teacher-forced in parallel via `fast_codebook_logits`. Raises on
  non-finite loss and on batches with zero supervised semantic frames.
- **Labels are pre-shifted** in `build_sft_example` (`input = full[:, :-1]`,
  `label = full[:, 1:]` with `-100` on prompt/padding) — the trainer does
  *not* shift again. Loss mask covers only the assistant/voice segment.
- **Prompt template** (no-reference path):
  `<|im_start|>system\nconvert the provided text to speech<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|voice|>`
  + target codes + `<|im_end|>`. The reference path prepends a
  `<|speaker:0|>{reference_text}` + reference-codes system block.
  **No language-ID token** — confirms §1.1.

### 6.6 Relevant defaults (`audio8_tts_sft.sh`)

| knob | default | note |
|---|---|---|
| `NPROC_PER_NODE` | 1 | single-GPU path; the L40 uses this |
| `BATCH_SIZE` / `GRADIENT_ACCUMULATION_STEPS` | 2 / 8 | effective batch 16 |
| `LEARNING_RATE` | 1e-5 | low, as expected for a pretrained AR |
| `MAX_LENGTH` | 2048 | = model `max_seq_len` cap; enforced |
| `BF16` / `GRADIENT_CHECKPOINTING` | true / true | |
| `FREEZE_SLOW_AR` / `FREEZE_FAST_AR` | false / false | isolate a branch |
| `NUM_TRAIN_EPOCHS` | 1 | |
| deepspeed | `configs/deepspeed_zero2.json` | |

Output saves to `output_dir` (and an optional clean `export_dir`), both
loadable via `AutoModel`/`AutoProcessor` with `trust_remote_code=True`.

### 6.7 Freeze flags (`set_trainable_modules`)

Slow AR = prefixes `embeddings. codebook_embeddings. layers. norm.`; fast AR
= `fast_project_in. fast_embeddings. fast_layers. fast_norm. fast_output.`.
Errors if both are frozen (zero trainable params). For a new-language
full fine-tune on the L40, keep both unfrozen.

### 6.8 Length filtering — important for the Basque recipe

`build_sft_example` **rejects** (raises, no truncation) any example whose
`text + [10,T] codes` exceeds `MAX_LENGTH=2048` positions. At ~21.5 codec
frames/s, 2048 positions ≈ a ~95s hard ceiling — but the §1.2 EOS-instability
signal argues for much shorter clips (≤ ~15–20s). Pre-filter by duration in
the manifest builder (the scaffold's `--max-duration-sec` does this via
ffprobe) rather than discovering rejections at `__getitem__` time.

---

## 7. Basque manifest builder (scaffold)

A scaffold was built in this repo (`zortzi-tts/`) to produce raw manifests in
the exact §6.2 schema, supplying the text normalization the upstream lacks.

### 7.1 Text normalizer (`basque_manifest/normalize.py`) — the centerpiece

- **Vigesimal number-to-words** (`2026` → `bi mila eta hogeita sei`,
  `21` → `hogeita bat`, `30` → `hogeita hamar`, `99` → `laurogeita hemeretzi`),
  with the "eta" connector before the final non-empty group
  (`101` → `ehun eta bat`, `1100` → `mila eta ehun`, `122` → `ehun eta hogeita bi`).
  Tables cross-checked against Euskaltzaindia / Omniglot.
- **Decimals** (`3,14` → `hiru koma bat lau`), **thousands separators**
  (`1.000` → `mila`, `1.000.000` → `milioi bat`, via 3-digit-grouping heuristic),
  **percent** (`50%` and `%50` → `ehuneko berrogeita hamar`), **currency**
  (`5€` → `bost euro`, `$10` → `dolar hamar`), **degrees** (`20°` → `hogei gradu`).
- **Preserves attached Basque suffixes** (`2026an` → `...hogeita seian`) —
  a heuristic with a known epenthesis edge case for digit-adjacent case
  endings (flagged TODO, see §4).
- **72 tests pass** (`uv run pytest`), covering the number table end-to-end
  and mixed-symbol sentences.

### 7.2 Sources & CLI

- `from-common-voice` — parses a CV eu export (`validated.tsv` + `clips/`),
  filters by up/down-votes, uses the clip stem as the id.
- `from-transcripts` — generic TSV/CSV loader with configurable id/text/audio
  columns; covers OpenSLR SLR76 and HiTZ Maider/Antton once the column layout
  is identified (avoids hard-coding layouts I have not yet downloaded).
- `merge` (dedup by id), `normalize` (re-normalize an existing manifest's text
  in place — useful for HiTZ), `stats`.
- Optional `--max-duration-sec` ffprobe pre-filter (the §6.8 length guard).
- Optional `--fixed-reference-audio/--fixed-reference-text` to attach one
  zero-shot reference voice to every record of a subset.

### 7.3 Status

Functional scaffold. The normalizer and both loaders are working end-to-end
(verified with a smoke test producing a valid `train.jsonl`). Open items
are the §4 native-speaker audit and the SLR76/HiTZ on-disk layout
confirmation — neither blocks building manifests from Common Voice today.
