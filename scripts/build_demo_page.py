#!/usr/bin/env python3
"""Generate the Zortzi-TTS demo website (docs/index.html).

Reads timing JSON files and produces a complete, self-contained HTML page
with an audio comparison table, model statistics, and academic styling.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

# ── Sentence definitions ────────────────────────────────────────────────
SENTENCES = [
    {
        "id": "s1",
        "type": "Greeting",
        "eu": "Kaixo, ona goiza denoi.",
        "en": "Hello, good morning everyone.",
    },
    {
        "id": "s2",
        "type": "Wh-Question",
        "eu": "Nondik zatoz zu?",
        "en": "Where do you come from?",
    },
    {
        "id": "s3",
        "type": "Yes/No Question",
        "eu": "Ba al dakizu euskaraz hitz egiten?",
        "en": "Do you know how to speak Basque?",
    },
    {
        "id": "s4",
        "type": "Exclamation",
        "eu": "Zein polita dago gaur eguzkia!",
        "en": "How beautiful the sun is today!",
    },
    {
        "id": "s5",
        "type": "Narrative",
        "eu": "Euskara Europako hizkuntzarik zaharrenetako bat da, eta milaka urteko historia du.",
        "en": "Basque is one of the oldest languages in Europe, with thousands of years of history.",
    },
    {
        "id": "s6",
        "type": "Mixed (Decl. + Question)",
        "eu": "Atzo Bilbora joan nintzen, eta zu non bizi zara?",
        "en": "Yesterday I went to Bilbao, and where do you live?",
    },
]

# ── Model definitions ───────────────────────────────────────────────────
MODELS = [
    {
        "id": "pytorch",
        "name": "Zortzi-TTS",
        "subtitle": "PyTorch · GPU",
        "arch": "Audio8-TTS 0.6B (fine-tuned)",
        "params": "601M",
        "size": "2.5 GB",
        "precision": "BF16",
        "sample_rate": "44.1 kHz",
        "rtf": "0.95",
        "hardware": "NVIDIA L40",
        "realtime": True,
        "voice_method": "Reference clip",
        "license": "Apache 2.0",
        "hf_url": "https://huggingface.co/itzune/zortzi-tts",
    },
    {
        "id": "onnx_fp16",
        "name": "Zortzi-TTS",
        "subtitle": "ONNX FP16 · CPU",
        "arch": "Audio8-TTS 0.6B (exported)",
        "params": "601M",
        "size": "1.5 GB",
        "precision": "FP16",
        "sample_rate": "44.1 kHz",
        "rtf": "5.4",
        "hardware": "CPU (8-core)",
        "realtime": False,
        "voice_method": "Reference clip",
        "license": "Apache 2.0",
        "hf_url": "https://huggingface.co/itzune/zortzi-tts-onnx",
    },
    {
        "id": "onnx_int4",
        "name": "Zortzi-TTS",
        "subtitle": "ONNX INT4 · CPU",
        "arch": "Audio8-TTS 0.6B (quantized)",
        "params": "601M",
        "size": "578 MB",
        "precision": "INT4",
        "sample_rate": "44.1 kHz",
        "rtf": "4.4",
        "hardware": "CPU (8-core)",
        "realtime": False,
        "voice_method": "Reference clip",
        "license": "Apache 2.0",
        "hf_url": "https://huggingface.co/itzune/zortzi-tts-onnx",
    },
    {
        "id": "piper",
        "name": "Piper",
        "subtitle": "VITS · CPU",
        "arch": "Piper VITS (medium)",
        "params": "15.7M",
        "size": "61 MB / voice",
        "precision": "FP32",
        "sample_rate": "22.05 kHz",
        "rtf": "0.07",
        "hardware": "CPU (8-core)",
        "realtime": True,
        "voice_method": "Per-voice model",
        "license": "MIT",
        "hf_url": "https://huggingface.co/itzune/maider-tts",
    },
]

VOICES = ["maider", "antton"]

# ── CSS ─────────────────────────────────────────────────────────────────
CSS = r"""
:root {
  --green: #009447;
  --green-dark: #007a3a;
  --red: #DA1F26;
  --bg: #ffffff;
  --bg-alt: #f7f8fa;
  --text: #1a1a2e;
  --text-muted: #6c757d;
  --border: #e2e8f0;
  --card-bg: #ffffff;
  --shadow: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
  --radius: 10px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: var(--text);
  background: var(--bg);
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--green); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Hero */
.hero {
  background: linear-gradient(135deg, #0a2e1a 0%, #0d4a2a 40%, #009447 100%);
  color: #fff;
  padding: 4rem 2rem 3.5rem;
  text-align: center;
}
.hero h1 {
  font-size: 3rem;
  font-weight: 800;
  letter-spacing: -1px;
  margin-bottom: .3rem;
}
.hero .tagline {
  font-size: 1.25rem;
  opacity: .9;
  font-weight: 300;
  margin-bottom: 1.5rem;
}
.hero .badges {
  display: flex;
  gap: .6rem;
  justify-content: center;
  flex-wrap: wrap;
}
.hero .badge {
  background: rgba(255,255,255,.15);
  border: 1px solid rgba(255,255,255,.25);
  border-radius: 999px;
  padding: .35rem 1rem;
  font-size: .85rem;
  font-weight: 500;
}
.hero .links {
  margin-top: 1.5rem;
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}
.hero .links a {
  color: #fff;
  background: rgba(255,255,255,.1);
  border: 1px solid rgba(255,255,255,.3);
  border-radius: 8px;
  padding: .5rem 1.2rem;
  font-size: .9rem;
  font-weight: 500;
  transition: background .2s;
}
.hero .links a:hover { background: rgba(255,255,255,.2); text-decoration: none; }

/* Sections */
.section {
  max-width: 1200px;
  margin: 0 auto;
  padding: 3rem 2rem;
}
.section h2 {
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 1rem;
  letter-spacing: -.5px;
}
.section h2 .anchor { color: var(--green); }
.section p { color: var(--text-muted); margin-bottom: 1rem; }

/* Features */
.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1.2rem;
  margin-top: 1.5rem;
}
.feature-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  box-shadow: var(--shadow);
}
.feature-card .icon { font-size: 1.75rem; margin-bottom: .5rem; }
.feature-card h3 { font-size: 1.05rem; font-weight: 600; margin-bottom: .3rem; }
.feature-card p { font-size: .9rem; margin: 0; }

/* Stats table */
.stats-wrapper { overflow-x: auto; margin-top: 1.5rem; }
.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: .92rem;
  min-width: 700px;
}
.stats-table th, .stats-table td {
  padding: .75rem 1rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.stats-table thead th {
  background: var(--bg-alt);
  font-weight: 600;
  font-size: .85rem;
  text-transform: uppercase;
  letter-spacing: .5px;
  color: var(--text-muted);
}
.stats-table tbody tr:hover { background: var(--bg-alt); }
.stats-table .metric { font-weight: 600; color: var(--text); }
.stats-table .yes { color: var(--green); font-weight: 600; }
.stats-table .no { color: var(--red); font-weight: 600; }
.stats-table .col-highlight { border-left: 3px solid var(--green); }

/* Audio table */
.audio-wrapper { overflow-x: auto; margin-top: 1.5rem; }
.audio-table {
  width: 100%;
  border-collapse: collapse;
  min-width: 900px;
}
.audio-table thead th {
  background: var(--bg-alt);
  padding: 1rem .75rem;
  text-align: center;
  border-bottom: 2px solid var(--border);
  font-weight: 600;
  font-size: .9rem;
}
.audio-table thead th .model-name { display: block; }
.audio-table thead th .model-sub {
  display: block;
  font-size: .75rem;
  font-weight: 400;
  color: var(--text-muted);
  margin-top: .15rem;
}
.audio-table thead th.sentence-col { text-align: left; min-width: 260px; }
.audio-table tbody td {
  padding: 1.25rem .75rem;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
  text-align: center;
}
.audio-table tbody tr:hover { background: #fafbfc; }

.sentence-type {
  display: inline-block;
  background: var(--green);
  color: #fff;
  font-size: .72rem;
  font-weight: 600;
  padding: .15rem .6rem;
  border-radius: 999px;
  margin-bottom: .5rem;
  text-transform: uppercase;
  letter-spacing: .5px;
}
.sentence-eu {
  font-size: .95rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: .15rem;
}
.sentence-en {
  font-size: .82rem;
  color: var(--text-muted);
  font-style: italic;
}

.voice-block {
  margin-bottom: .75rem;
}
.voice-block:last-child { margin-bottom: 0; }
.voice-label {
  display: block;
  font-size: .72rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .5px;
  margin-bottom: .2rem;
}
.voice-label.female { color: #c2185b; }
.voice-label.male { color: #1565c0; }
.audio-table audio {
  width: 100%;
  min-width: 160px;
  height: 32px;
}

/* Methodology */
.method-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-top: 1.5rem;
}
.method-card {
  background: var(--bg-alt);
  border-radius: var(--radius);
  padding: 1.5rem;
  border-left: 4px solid var(--green);
}
.method-card h3 { font-size: 1rem; font-weight: 600; margin-bottom: .5rem; }
.method-card p { font-size: .9rem; margin: 0; }
.method-card .stat {
  display: inline-block;
  background: var(--green);
  color: #fff;
  font-size: .8rem;
  font-weight: 600;
  padding: .2rem .7rem;
  border-radius: 6px;
  margin-top: .5rem;
}

/* Footer */
footer {
  background: #0a2e1a;
  color: rgba(255,255,255,.8);
  padding: 2.5rem 2rem;
  text-align: center;
}
footer .footer-links {
  display: flex;
  gap: 1.5rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}
footer a { color: rgba(255,255,255,.9); }
footer p { font-size: .85rem; color: rgba(255,255,255,.6); }
footer .citation {
  text-align: left;
  max-width: 700px;
  margin: 1.5rem auto 0;
  background: rgba(255,255,255,.05);
  border-radius: 8px;
  padding: 1rem;
  font-size: .8rem;
  font-family: monospace;
  white-space: pre-wrap;
  color: rgba(255,255,255,.7);
}

@media (max-width: 768px) {
  .hero h1 { font-size: 2rem; }
  .hero .tagline { font-size: 1rem; }
  .section { padding: 2rem 1rem; }
}
"""

# ── HTML generation ────────────────────────────────────────────────────

def generate_stats_table() -> str:
    metrics = [
        ("Architecture", "arch"),
        ("Parameters", "params"),
        ("Model Size", "size"),
        ("Precision", "precision"),
        ("Sample Rate", "sample_rate"),
        ("Avg RTF", "rtf"),
        ("Hardware", "hardware"),
        ("Real-time?", "realtime"),
        ("Voice Method", "voice_method"),
        ("License", "license"),
    ]
    rows = ""
    for label, key in metrics:
        cells = f'<td class="metric">{label}</td>'
        for m in MODELS:
            val = m[key]
            if key == "realtime":
                val = '<span class="yes">✓ Yes</span>' if val else '<span class="no">✗ No</span>'
            cells += f"<td>{val}</td>"
        rows += f"<tr>{cells}</tr>\n"

    headers = "<th></th>"
    for m in MODELS:
        highlight = " col-highlight" if m["id"] == "pytorch" else ""
        headers += f'<th class="{highlight}">{m["name"]}<br><span class="model-sub">{m["subtitle"]}</span></th>'

    return f"""
<table class="stats-table">
  <thead><tr>{headers}</tr></thead>
  <tbody>
{rows}
  </tbody>
</table>"""


def generate_audio_table() -> str:
    headers = '<th class="sentence-col">Sentence</th>'
    for m in MODELS:
        highlight = " col-highlight" if m["id"] == "pytorch" else ""
        headers += (
            f'<th class="{highlight}">'
            f'<span class="model-name">{m["name"]}</span>'
            f'<span class="model-sub">{m["subtitle"]}</span>'
            f'</th>'
        )

    rows = ""
    for s in SENTENCES:
        cells = (
            f'<td>'
            f'<span class="sentence-type">{s["type"]}</span>'
            f'<div class="sentence-eu">{s["eu"]}</div>'
            f'<div class="sentence-en">{s["en"]}</div>'
            f'</td>'
        )
        for m in MODELS:
            voice_cells = ""
            for v in VOICES:
                vclass = "female" if v == "maider" else "male"
                audio_path = f"audio/{m['id']}_{v}_{s['id']}.mp3"
                voice_cells += (
                    f'<div class="voice-block">'
                    f'<span class="voice-label {vclass}">{v.title()}</span>'
                    f'<audio controls preload="none" src="{audio_path}"></audio>'
                    f'</div>'
                )
            cells += f'<td>{voice_cells}</td>'
        rows += f"<tr>{cells}</tr>\n"

    return f"""
<div class="audio-wrapper">
<table class="audio-table">
  <thead><tr>{headers}</tr></thead>
  <tbody>
{rows}
  </tbody>
</table>
</div>"""


def generate_html() -> str:
    stats = generate_stats_table()
    audio = generate_audio_table()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Zortzi-TTS · Basque Text-to-Speech Demo</title>
  <meta name="description" content="Zortzi-TTS: A Basque text-to-speech system fine-tuned from Audio8-TTS-0.6B with natural prosody, supporting Maider and Antton voices.">
  <style>
{CSS}
  </style>
</head>
<body>

<!-- ═══ Hero ═══ -->
<header class="hero">
  <h1>Zortzi-TTS</h1>
  <p class="tagline">Basque Text-to-Speech with Natural Prosody</p>
  <div class="badges">
    <span class="badge">🎭 Two voices: Maider &amp; Antton</span>
    <span class="badge">🧠 Fine-tuned from Audio8-TTS 0.6B</span>
    <span class="badge">📚 Trained on 175K+ Basque clips</span>
    <span class="badge">📜 Apache 2.0</span>
  </div>
  <div class="links">
    <a href="https://huggingface.co/itzune/zortzi-tts">📦 PyTorch Model</a>
    <a href="https://huggingface.co/itzune/zortzi-tts-onnx">📦 ONNX Model</a>
    <a href="https://github.com/itzune/zortzi-tts">💻 GitHub</a>
  </div>
</header>

<!-- ═══ Overview ═══ -->
<section class="section" id="overview">
  <h2><span class="anchor">§</span> Overview</h2>
  <p>
    <strong>Zortzi-TTS</strong> is a Basque text-to-speech system fine-tuned from the
    <a href="https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b">Audio8-TTS-0.6B</a> preview model.
    It was trained using a two-phase curriculum on Mozilla Common Voice 26.0 (Basque, 134K clips)
    and the HiTZ Aholab TTS corpus, producing two natural-sounding voices:
    <strong>Maider</strong> (female) and <strong>Antton</strong> (male).
  </p>
  <p>
    The model captures Basque-specific phonotactics and prosody, including correct rising intonation
    on yes/no questions, focal pitch peaks on wh-question words, and expressive exclamatory contours.
    On this page we compare four deployment configurations: the original PyTorch model (GPU),
    ONNX FP16 and INT4 exports (CPU), and Piper TTS (a lightweight VITS baseline trained on the same voices).
  </p>
</section>

<!-- ═══ Features ═══ -->
<section class="section" id="features" style="background:var(--bg-alt);">
  <h2><span class="anchor">§</span> Key Features</h2>
  <div class="features-grid">
    <div class="feature-card">
      <div class="icon">🗣️</div>
      <h3>Two Natural Voices</h3>
      <p>Maider (female) and Antton (male), anchored from HiTZ Aholab professional recordings.</p>
    </div>
    <div class="feature-card">
      <div class="icon">🎵</div>
      <h3>Correct Basque Prosody</h3>
      <p>Rising intonation on yes/no questions, focus peaks on wh-words, and expressive exclamations.</p>
    </div>
    <div class="feature-card">
      <div class="icon">⚡</div>
      <h3>Multiple Deploy Targets</h3>
      <p>PyTorch (GPU), ONNX FP16, and ONNX INT4 (578 MB) for CPU deployment — no GPU required.</p>
    </div>
    <div class="feature-card">
      <div class="icon">📚</div>
      <h3>Large Training Corpus</h3>
      <p>134K Common Voice + 41K HiTZ clips (with 3× upsampled interrogatives/exclamatives).</p>
    </div>
    <div class="feature-card">
      <div class="icon">🎛️</div>
      <h3>44.1 kHz Output</h3>
      <p>Full-band audio at CD-quality sample rate, far surpassing typical 22 kHz neural TTS.</p>
    </div>
    <div class="feature-card">
      <div class="icon">🔧</div>
      <h3>Reproducible Pipeline</h3>
      <p>Complete manifest builder, training scripts, and ONNX export toolchain included.</p>
    </div>
  </div>
</section>

<!-- ═══ Model Comparison ═══ -->
<section class="section" id="comparison">
  <h2><span class="anchor">§</span> Model Comparison</h2>
  <p>
    RTF (Real-Time Factor) = inference_time ÷ audio_duration. Values below 1.0 indicate
    faster-than-real-time synthesis. All CPU measurements use an 8-core server; the PyTorch
    model runs on an NVIDIA L40 GPU.
  </p>
  <div class="stats-wrapper">
{stats}
  </div>
</section>

<!-- ═══ Audio Samples ═══ -->
<section class="section" id="samples" style="background:var(--bg-alt);">
  <h2><span class="anchor">§</span> Audio Samples</h2>
  <p>
    Six sentences covering greetings, questions, exclamations, narratives, and mixed utterances.
    Each cell contains two audio players — one for each voice. Click play to compare; playing one
    will automatically pause others.
  </p>
{audio}
</section>

<!-- ═══ Methodology ═══ -->
<section class="section" id="methodology">
  <h2><span class="anchor">§</span> Training Methodology</h2>
  <div class="method-grid">
    <div class="method-card">
      <h3>Phase 1 — Phonotactics</h3>
      <p>Common Voice 26.0 Basque (134K clips). Trains the model on Basque sound patterns
      without reference audio. Stopped at step 6,000 to preserve prosodic flexibility.</p>
      <span class="stat">6,000 steps · 64 min</span>
    </div>
    <div class="method-card">
      <h3>Phase 2 — Voice + Prosody</h3>
      <p>HiTZ Aholab corpus (41K clips) with 3× upsampled interrogatives and exclamatives.
      Anchors Maider &amp; Antton voices while polishing question intonation.</p>
      <span class="stat">7,746 steps · 98 min</span>
    </div>
    <div class="method-card">
      <h3>ONNX Export &amp; Quantization</h3>
      <p>Custom ONNX export (opset 17) with manual attention and KV-cache. Two-pass INT4
      quantization: MatMulNBits (weights) + GatherBlockQuantized (embeddings).</p>
      <span class="stat">578 MB total · INT4</span>
    </div>
    <div class="method-card">
      <h3>Piper Baseline</h3>
      <p>Separately trained Piper VITS medium models for the same voices. Lightweight (15.7M params)
      and extremely fast on CPU, but with lower audio quality (22 kHz).</p>
      <span class="stat">15.7M params · 61 MB</span>
    </div>
  </div>
</section>

<!-- ═══ Footer ═══ -->
<footer>
  <div class="footer-links">
    <a href="https://huggingface.co/itzune/zortzi-tts">HuggingFace (PyTorch)</a>
    <a href="https://huggingface.co/itzune/zortzi-tts-onnx">HuggingFace (ONNX)</a>
    <a href="https://github.com/itzune/zortzi-tts">GitHub</a>
    <a href="https://github.com/itzune/zortzi-tts#training">Training Docs</a>
  </div>
  <p>Zortzi-TTS · Fine-tuned from Audio8-TTS-0.6B · Apache 2.0 License</p>
  <p>Trained on Mozilla Common Voice 26.0 (CC0) and HiTZ Aholab TTS (CC BY 4.0)</p>
  <div class="citation">@misc{{zortzi-tts,
  title  = {{Zortzi-TTS: Basque Text-to-Speech with Natural Prosody}},
  author = {{Itzune}},
  year   = {{2025}},
  note   = {{Fine-tuned from Audio8-TTS-0.6B. Trained on Common Voice 26.0 and HiTZ Aholab TTS.}},
  url    = {{https://github.com/itzune/zortzi-tts}}
}}</div>
</footer>

<script>
  // Pause all other audio elements when one starts playing
  document.querySelectorAll('audio').forEach(function(a) {{
    a.addEventListener('play', function() {{
      document.querySelectorAll('audio').forEach(function(b) {{
        if (b !== a) b.pause();
      }});
    }});
  }});
</script>

</body>
</html>"""


if __name__ == "__main__":
    html = generate_html()
    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Generated {out} ({len(html):,} bytes)")
