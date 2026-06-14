---
title: Rivet R Coach
emoji: 🌀
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.50.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
hf_oauth: true
short_description: On-device English /r/ speech coach
tags:
  - backyard-ai
  - off-brand
  - tiny-titan
  - best-demo
  - nemotron
---

# 🌀 Rivet R Coach

A **practical, on-device speech-therapy coach** for the English /r/ sound — the most-mispronounced consonant in American English and the #1 reason kids get sent to speech therapy.

Open the app, say a word into your mic, and a real audio pipeline tells you — instantly — whether your "R" actually landed and *exactly* what to move next.

🎥 **Demo video:** <!-- TODO: paste YouTube/Loom link --> `<DEMO_VIDEO_URL>`
🐦 **Social post:** <!-- TODO: paste X/LinkedIn link --> `<SOCIAL_POST_URL>`
🧑‍💻 **GitHub:** https://github.com/IndianChess/rhotacism-nemotron-coach

---

## The idea

About 1 in 20 kids — and a long tail of adults — can't make a clean American /r/. Speech therapy works, but it's expensive and most practice happens at home with zero feedback. Rivet R Coach is the "alone-at-home-with-feedback" part: small models, real audio analysis, and a warm coach that runs on your laptop.

## How it works

```
🎤 your voice
   ├── wav2vec2 phoneme model  →  IPA transcription
   ├── Praat (parselmouth)     →  F3 formant in Hz (the acoustic fingerprint of /r/)
   ├── scoring.py              →  correct / approaching / w-substitution / omission / distortion
   └── Nemotron-3-Nano-4B      →  spoken feedback ("Curl your tongue further back…")
                                ↓
                       🔊 pocket-tts (on-device voice)
```

Every model in the pipeline runs **locally**. No mic audio ever leaves the machine; in `local` coach mode the LLM stays on-device too.

## Models used (all ≤32B ✓)

| Component | Model | Params | Where |
|---|---|---|---|
| Phoneme transcription | `vitouphy/wav2vec2-xls-r-300m-phoneme` | ~315 M | local |
| Coach LLM (default) | `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` (Q4_K_M) | 4 B | local (llama.cpp) |
| Coach LLM (router fallback) | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | 8 B | HF Inference Providers |
| Voice (TTS) | `pocket-tts` | ~250 M | local |

Combined active parameter count: well under the 32 B cap, in either mode.

## Features

- **Five curriculum levels:** Syllables → Starting Words → Middle/End → Phrases → Twisters
- **Daily tongue twister** (rotates by date — same one for everyone, every day)
- **Streaks, XP, level-ups** with auto-advance after 5 consecutive correct
- **Custom UI** — no default Gradio chrome; sidebar nav, level path, blob coach character, profile view
- **HF OAuth + persistent progress** via a private HF Dataset (`IndianChess/rivet-progress`)
- **Two coach backends** — `local` (llama.cpp, fully offline) or `router` (HF Inference Providers)

## Run locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# espeak-ng is a system dep for phonemizer
brew install espeak-ng              # macOS
# sudo apt-get install espeak-ng    # Debian/Ubuntu

cp .env.example .env                # optional: add HF_TOKEN for router mode
.venv/bin/python app.py             # opens on http://127.0.0.1:7860
```

> ⏳ **First boot downloads ~3 GB** — wav2vec2 (~315 MB) + Nemotron GGUF (~2.84 GB) + pocket-tts (~250 MB) cache into `~/.cache/huggingface/hub/`. Subsequent boots are instant.

## Coach backend

| `COACH_BACKEND` | Model | Runtime | Needs |
|---|---|---|---|
| `local` *(default)* | `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` (Q4_K_M, 2.84 GB) | llama-cpp-python, Metal/CUDA/CPU auto | — |
| `router` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | HF Inference Providers (featherless-ai) | `HF_TOKEN` |

On Apple Silicon, local coach turns land in ~1–2 s. On a free HF Space CPU tier, expect ~20–40 s per coach turn — the demo video shows the M-series experience.

Override the GGUF source if you want a different quant:

```bash
COACH_BACKEND=local \
COACH_LOCAL_REPO=lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF \
COACH_LOCAL_QUANT=Q6_K \
python app.py
```

## Deploy on Hugging Face Spaces

1. Create a new Space inside the **Build Small hackathon org** with `SDK = Gradio`.
2. Push this repo to the Space's git remote.
3. **Add `HF_TOKEN` as a Space secret** (Settings → Variables and secrets):
   - Get a read token at <https://huggingface.co/settings/tokens>
   - Required for OAuth progress sync and `router` coach mode. The app will boot without it; coach falls back to a generic message and progress is session-only.

## Building your own word list

`words.py` ships with dictionary IPA. For sharper phoneme matching, drop your own reference recordings into `recordings/<word>.wav` and run:

```bash
python build_word_list.py
```

It transcribes each recording with the same wav2vec2 model the app uses and lets you confirm or edit the IPA before saving.

## Disclaimer

This is a practice tool, not a substitute for a licensed speech-language pathologist. Recordings are processed locally; in `router` mode the score summary (not the audio) is sent to a hosted LLM for feedback generation.

---

Built for the **Hugging Face × Gradio Build Small Hackathon**, June 2026 · Backyard AI track.
