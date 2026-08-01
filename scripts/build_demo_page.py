#!/usr/bin/env python3
"""Generate the Zortzi-TTS demo website (docs/index.html).

Uses the Itzune design system: dark cosmic-void background, accent palette
sourced from the Itzune logo (sky, magenta, orange, amber, purple),
DM Sans typography, minimalist academic layout. No emojis.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

# ── Load timing data ────────────────────────────────────────────────────
def load_timing(path: Path) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        return [s for r in d for s in r["samples"]]
    return d["samples"]

pytorch_samples = load_timing(ROOT / "probes" / "demo_server" / "timing_pytorch.json")
onnx_raw = json.loads((ROOT / "probes" / "demo_server" / "timing_onnx.json").read_text())
onnx_fp16_samples = onnx_raw[0]["samples"]
onnx_int4_samples = onnx_raw[1]["samples"]
piper_samples = load_timing(ROOT / "probes" / "demo_piper" / "timing_piper.json")


def median_rtf(samples: list[dict]) -> float:
    return statistics.median(s["rtf"] for s in samples)


# ── Sentence definitions (corrected Basque) ─────────────────────────────
SENTENCES = [
    {"id": "s1", "type": "Greeting",        "eu": "Kaixo, egun on guztioi.",                     "en": "Hello, good morning everyone."},
    {"id": "s2", "type": "Wh-question",     "eu": "Nondik zatoz zu?",                             "en": "Where do you come from?"},
    {"id": "s3", "type": "Yes/no question", "eu": "Ba al dakizu euskaraz hitz egiten?",           "en": "Do you know how to speak Basque?"},
    {"id": "s4", "type": "Exclamation",     "eu": "Zein polita dagoen gaur eguzkia!",             "en": "How beautiful the sun is today!"},
    {"id": "s5", "type": "Narrative",       "eu": "Euskara Europako hizkuntzarik zaharrenetako bat da, eta milaka urteko historia du.", "en": "Basque is one of the oldest languages in Europe, with thousands of years of history."},
    {"id": "s6", "type": "Declarative + question", "eu": "Ni Bilbon bizi naiz, eta zu non bizi zara?", "en": "I live in Bilbao, and where do you live?"},
]

# ── Model definitions ───────────────────────────────────────────────────
MODELS = [
    {
        "id": "pytorch",
        "name": "Zortzi-TTS",
        "subtitle": "PyTorch / GPU",
        "arch": "Audio8-TTS 0.6B (fine-tuned)",
        "params": "601M",
        "size": "2.5 GB",
        "precision": "BF16",
        "sample_rate": "44.1 kHz",
        "rtf": f"{median_rtf(pytorch_samples):.2f}",
        "hardware": "NVIDIA L40",
        "realtime": True,
        "voice_method": "Reference clip",
        "license": "Apache 2.0",
        "hf_url": "https://huggingface.co/itzune/zortzi-tts",
    },
    {
        "id": "onnx_fp16",
        "name": "Zortzi-TTS",
        "subtitle": "ONNX FP16 / CPU",
        "arch": "Audio8-TTS 0.6B (exported)",
        "params": "601M",
        "size": "1.5 GB",
        "precision": "FP16",
        "sample_rate": "44.1 kHz",
        "rtf": f"{median_rtf(onnx_fp16_samples):.1f}",
        "hardware": "CPU (8-core)",
        "realtime": False,
        "voice_method": "Reference clip",
        "license": "Apache 2.0",
        "hf_url": "https://huggingface.co/itzune/zortzi-tts-onnx",
    },
    {
        "id": "onnx_int4",
        "name": "Zortzi-TTS",
        "subtitle": "ONNX INT4 / CPU",
        "arch": "Audio8-TTS 0.6B (quantized)",
        "params": "601M",
        "size": "578 MB",
        "precision": "INT4",
        "sample_rate": "44.1 kHz",
        "rtf": f"{median_rtf(onnx_int4_samples):.1f}",
        "hardware": "CPU (8-core)",
        "realtime": False,
        "voice_method": "Reference clip",
        "license": "Apache 2.0",
        "hf_url": "https://huggingface.co/itzune/zortzi-tts-onnx",
    },
    {
        "id": "piper",
        "name": "Piper",
        "subtitle": "VITS / CPU",
        "arch": "Piper VITS (medium)",
        "params": "15.7M",
        "size": "61 MB / voice",
        "precision": "FP32",
        "sample_rate": "22.05 kHz",
        "rtf": f"{median_rtf(piper_samples):.2f}",
        "hardware": "CPU (8-core)",
        "realtime": True,
        "voice_method": "Per-voice model",
        "license": "MIT",
        "hf_url": "https://huggingface.co/itzune/maider-tts",
    },
]

VOICES = ["maider", "antton"]

# ── CSS (Itzune design system) ──────────────────────────────────────────
CSS = r"""
:root {
  /* Surfaces */
  --cosmic-void: #06051d;
  --deep-navy:   #061434;
  --abyssal-blue: #0f1c36;
  --steel-navy:  #1d293d;
  --deep-slate:  #314062;

  /* Neutrals */
  --mist:        #cad5e2;
  --fog:         #e5e7eb;
  --ghost-white: #ffffff;
  --ice-blue:    #ebf8ff;

  /* Itzune accent palette (from logo) */
  --itzune-sky:     #4bb8e8;
  --itzune-magenta: #e91e8c;
  --itzune-orange:  #f07030;
  --itzune-amber:   #f5a020;
  --itzune-purple:  #8b35b8;

  /* Card */
  --card-bg: rgba(255, 255, 255, 0.04);
  --card-border: rgba(255, 255, 255, 0.08);
  --card-border-hover: rgba(255, 255, 255, 0.15);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: "DM Sans", system-ui, sans-serif;
  color: var(--mist);
  background: var(--cosmic-void);
  background-image:
    radial-gradient(circle at 15% 15%, rgba(139, 53, 184, 0.12), transparent 55%),
    radial-gradient(circle at 85% 5%, rgba(233, 30, 140, 0.08), transparent 50%),
    radial-gradient(circle at 30% 90%, rgba(75, 184, 232, 0.08), transparent 60%);
  background-attachment: fixed;
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
}

a { color: var(--itzune-sky); text-decoration: none; transition: color .2s; }
a:hover { color: var(--ice-blue); }

/* ─── Header ─── */
header.nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 2rem;
  max-width: 1200px;
  margin: 0 auto;
}
.nav-left {
  display: flex;
  align-items: center;
  gap: .6rem;
}
.nav-logo {
  width: 28px;
  height: 28px;
  border-radius: 6px;
  opacity: 0.9;
}
.nav-brand {
  font-family: "Space Grotesk", system-ui, sans-serif;
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--ghost-white);
  letter-spacing: -.5px;
}
.nav-right {
  display: flex;
  align-items: center;
  gap: .75rem;
}
.nav-pill {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  padding: .35rem .9rem;
  border: 1px solid var(--card-border);
  border-radius: 9999px;
  font-size: .82rem;
  font-weight: 500;
  color: var(--mist);
  transition: border-color .2s, color .2s, background .2s;
}
.nav-pill:hover {
  border-color: var(--itzune-sky);
  color: var(--ice-blue);
  background: rgba(75, 184, 232, 0.08);
  text-decoration: none;
}
.nav-pill svg { width: 14px; height: 14px; }

/* ─── Hero ─── */
.hero {
  text-align: center;
  padding: 3rem 2rem 2rem;
  max-width: 820px;
  margin: 0 auto;
}
.hero h1 {
  font-family: "Space Grotesk", system-ui, sans-serif;
  font-size: 2.8rem;
  font-weight: 700;
  color: var(--ghost-white);
  letter-spacing: -1px;
  margin-bottom: .4rem;
}
.hero .tagline {
  font-size: 1.15rem;
  color: var(--mist);
  font-weight: 300;
  margin-bottom: 1.5rem;
}
.hero .badges {
  display: flex;
  gap: .5rem;
  justify-content: center;
  flex-wrap: wrap;
}
.hero .badge {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 9999px;
  padding: .3rem .85rem;
  font-size: .8rem;
  font-weight: 500;
  color: var(--fog);
}
.hero .badge .dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  margin-right: .35rem;
  vertical-align: middle;
}
.dot-sky     { background: var(--itzune-sky); }
.dot-magenta { background: var(--itzune-magenta); }
.dot-amber   { background: var(--itzune-amber); }
.dot-purple  { background: var(--itzune-purple); }

/* ─── Sections ─── */
.section {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.5rem 2rem;
}
.section h2 {
  font-family: "Space Grotesk", system-ui, sans-serif;
  font-size: 1.4rem;
  font-weight: 600;
  color: var(--ghost-white);
  margin-bottom: .75rem;
  letter-spacing: -.3px;
}
.section p {
  color: var(--mist);
  margin-bottom: 1rem;
  font-size: .95rem;
}

/* ─── Stats table ─── */
.stats-wrapper { overflow-x: auto; margin-top: 1.5rem; }
.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: .88rem;
  min-width: 700px;
}
.stats-table th, .stats-table td {
  padding: .7rem 1rem;
  text-align: left;
  border-bottom: 1px solid var(--card-border);
}
.stats-table thead th {
  font-weight: 600;
  font-size: .8rem;
  text-transform: uppercase;
  letter-spacing: .5px;
  color: var(--mist);
  padding-bottom: 1rem;
}
.stats-table thead th .model-sub {
  display: block;
  font-size: .72rem;
  font-weight: 400;
  color: var(--itzune-sky);
  margin-top: .15rem;
  text-transform: none;
  letter-spacing: 0;
}
.stats-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.stats-table .metric { font-weight: 500; color: var(--fog); }
.stats-table .yes { color: var(--itzune-sky); font-weight: 600; }
.stats-table .no { color: var(--itzune-magenta); font-weight: 600; }
.stats-table .col-highlight {
  border-left: 2px solid var(--itzune-sky);
  background: rgba(75, 184, 232, 0.03);
}

/* ─── Audio table ─── */
.audio-wrapper { overflow-x: auto; margin-top: 1.5rem; }
.audio-table {
  width: 100%;
  border-collapse: collapse;
  min-width: 900px;
}
.audio-table thead th {
  padding: 1rem .75rem;
  text-align: center;
  border-bottom: 1px solid var(--card-border);
  font-weight: 600;
  font-size: .88rem;
  color: var(--ghost-white);
  vertical-align: bottom;
}
.audio-table thead th .model-sub {
  display: block;
  font-size: .72rem;
  font-weight: 400;
  color: var(--itzune-sky);
  margin-top: .15rem;
}
.audio-table thead th.sentence-col {
  text-align: left;
  min-width: 260px;
}
.audio-table thead th.col-highlight {
  border-left: 2px solid var(--itzune-sky);
}
.audio-table tbody td {
  padding: 1.1rem .75rem;
  border-bottom: 1px solid var(--card-border);
  vertical-align: top;
  text-align: center;
}
.audio-table tbody tr:hover { background: rgba(255,255,255,0.02); }

.sentence-type {
  display: inline-block;
  font-size: .68rem;
  font-weight: 600;
  padding: .15rem .55rem;
  border-radius: 4px;
  margin-bottom: .4rem;
  text-transform: uppercase;
  letter-spacing: .5px;
  border: 1px solid var(--card-border);
  color: var(--itzune-sky);
}
.sentence-eu {
  font-size: .92rem;
  font-weight: 500;
  color: var(--fog);
  margin-bottom: .1rem;
}
.sentence-en {
  font-size: .8rem;
  color: var(--mist);
  opacity: .7;
  font-style: italic;
}

.voice-block { margin-bottom: .6rem; }
.voice-block:last-child { margin-bottom: 0; }
.voice-label {
  display: block;
  font-size: .68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .5px;
  margin-bottom: .15rem;
}
.voice-label.female { color: var(--itzune-magenta); }
.voice-label.male   { color: var(--itzune-sky); }

/* Style native audio player for dark theme */
.audio-table audio {
  width: 100%;
  min-width: 150px;
  height: 30px;
  filter: invert(0.85) hue-rotate(180deg);
}

/* ─── Footer ─── */
footer {
  border-top: 1px solid var(--card-border);
  padding: 2rem;
  text-align: center;
  max-width: 1200px;
  margin: 2rem auto 0;
}
footer .footer-links {
  display: flex;
  gap: 1.2rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: .8rem;
}
footer a { font-size: .85rem; }
footer p { font-size: .8rem; color: var(--mist); opacity: .6; }

@media (max-width: 768px) {
  .hero h1 { font-size: 2rem; }
  .hero .tagline { font-size: 1rem; }
  .section { padding: 1.5rem 1rem; }
  header.nav { padding: .75rem 1rem; }
}
"""

# ── SVG icons (no emoji) ────────────────────────────────────────────────
ICON_GITHUB = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
ICON_HF = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M12 4.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zm-5 0a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM8 9a6 6 0 00-5.2 3h2.4A4 4 0 018 10a4 4 0 013 2h2.4A6 6 0 008 9zm0-9a8 8 0 100 16A8 8 0 008 0z"/></svg>'


# ── HTML generation ────────────────────────────────────────────────────

def generate_stats_table() -> str:
    metrics = [
        ("Architecture", "arch"),
        ("Parameters", "params"),
        ("Model size", "size"),
        ("Precision", "precision"),
        ("Sample rate", "sample_rate"),
        ("Median RTF", "rtf"),
        ("Hardware", "hardware"),
        ("Real-time", "realtime"),
        ("Voice method", "voice_method"),
        ("License", "license"),
    ]
    rows = ""
    for label, key in metrics:
        cells = f'<td class="metric">{label}</td>'
        for m in MODELS:
            val = m[key]
            if key == "realtime":
                val = '<span class="yes">Yes</span>' if val else '<span class="no">No</span>'
            highlight = " col-highlight" if m["id"] == "pytorch" else ""
            cells += f'<td class="{highlight.strip()}">{val}</td>'
        rows += f"<tr>{cells}</tr>\n"

    headers = "<th></th>"
    for m in MODELS:
        highlight = " col-highlight" if m["id"] == "pytorch" else ""
        headers += f'<th class="{highlight.strip()}">{m["name"]}<span class="model-sub">{m["subtitle"]}</span></th>'

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
        headers += f'<th class="{highlight.strip()}">{m["name"]}<span class="model-sub">{m["subtitle"]}</span></th>'

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
  <title>Zortzi-TTS — Basque Text-to-Speech</title>
  <meta name="description" content="Zortzi-TTS: a Basque text-to-speech system fine-tuned from Audio8-TTS-0.6B with natural prosody, supporting Maider and Antton voices.">
  <link rel="icon" href="itzune-logo.jpg" type="image/jpeg">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
{CSS}
  </style>
</head>
<body>

<!-- ═══ Header ═══ -->
<header class="nav">
  <a href="https://itzune.eus" class="nav-left">
    <img src="itzune-logo.jpg" alt="Itzune" class="nav-logo">
    <span class="nav-brand">itzune</span>
  </a>
  <div class="nav-right">
    <a href="https://github.com/itzune/zortzi-tts" class="nav-pill">{ICON_GITHUB} GitHub</a>
    <a href="https://huggingface.co/itzune" class="nav-pill">{ICON_HF} Hugging Face</a>
  </div>
</header>

<!-- ═══ Hero ═══ -->
<section class="hero">
  <h1>Zortzi-TTS</h1>
  <p class="tagline">Basque text-to-speech with natural prosody</p>
  <div class="badges">
    <span class="badge"><span class="dot dot-magenta"></span>Two voices: Maider &amp; Antton</span>
    <span class="badge"><span class="dot dot-sky"></span>Fine-tuned from Audio8-TTS 0.6B</span>
    <span class="badge"><span class="dot dot-amber"></span>175K+ Basque clips</span>
    <span class="badge"><span class="dot dot-purple"></span>Apache 2.0</span>
  </div>
</section>

<!-- ═══ Overview ═══ -->
<section class="section" id="overview">
  <h2>Overview</h2>
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
    This page compares four deployment configurations: the PyTorch model (GPU), ONNX FP16 and INT4
    exports (CPU), and Piper TTS (a lightweight VITS baseline trained on the same voices).
  </p>
</section>

<!-- ═══ Model Comparison ═══ -->
<section class="section" id="comparison">
  <h2>Model Comparison</h2>
  <p>
    RTF (Real-Time Factor) = inference time / audio duration. Values below 1.0 indicate
    faster-than-real-time synthesis. Median values are shown to reduce cold-start outliers.
  </p>
  <div class="stats-wrapper">
{stats}
  </div>
</section>

<!-- ═══ Audio Samples ═══ -->
<section class="section" id="samples">
  <h2>Audio Samples</h2>
  <p>
    Six sentences covering greetings, questions, exclamations, narratives, and mixed utterances.
    Each cell contains two audio players — one per voice. Playing one will automatically pause others.
  </p>
{audio}
</section>

<!-- ═══ Footer ═══ -->
<footer>
  <div class="footer-links">
    <a href="https://huggingface.co/itzune/zortzi-tts">HuggingFace (PyTorch)</a>
    <a href="https://huggingface.co/itzune/zortzi-tts-onnx">HuggingFace (ONNX)</a>
    <a href="https://github.com/itzune/zortzi-tts">GitHub</a>
    <a href="https://itzune.eus">itzune.eus</a>
  </div>
  <p>Fine-tuned from Audio8-TTS-0.6B — Apache 2.0 License</p>
  <p>Trained on Mozilla Common Voice 26.0 (CC0) and HiTZ Aholab TTS (CC BY 4.0)</p>
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
