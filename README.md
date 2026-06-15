---
title: Rhotic
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
  - codex
  - track:backyard
  - sponsor:openai
  - sponsor:nvidia
  - achievement:offgrid
  - achievement:offbrand
  - achievement:llama
  - achievement:fieldnotes
---

# 🌀 Rhotic

An on-device speech coach for the English /r/ sound.

You open the app, say a word into your mic, and get back two things: a real acoustic readout of whether your "R" landed, and a short spoken note telling you what to fix next.

**Track:** Backyard AI
**Sponsor prizes targeted:** NVIDIA Nemotron Hardware Prize, OpenAI Best Use of Codex

🎥 **Demo video:** <!-- TODO: paste YouTube/Loom link --> `<DEMO_VIDEO_URL>`
🐦 **Social post:** <!-- TODO: paste X/LinkedIn link --> `<SOCIAL_POST_URL>`
🧑‍💻 **GitHub:** https://github.com/IndianChess/rhotacism-nemotron-coach

---

## Why this exists

About 1 in 20 kids can't make a clean American /r/, and a lot of adults can't either. Speech therapy works, but it's expensive, and the part that actually moves the needle (daily practice at home) usually happens with zero feedback. Most home-practice apps just show flashcards. Rhotic is the missing feedback loop.

## How it works

```
🎤 your voice
   ├── wav2vec2 phoneme model  →  IPA transcription
   ├── Praat (parselmouth)     →  F3 formant in Hz (the acoustic fingerprint of /r/)
   ├── scoring.py              →  correct / approaching / w-substitution / omission / distortion
   └── Nemotron-3-Nano-4B      →  spoken feedback ("Curl your tongue further back...")
                                ↓
                       🔊 pocket-tts (on-device voice)
```

Every model runs locally. No mic audio leaves the machine. The LLM coach stays on-device too.

## Models used (all ≤32B ✓)

| Component | Model | Params | Where |
|---|---|---|---|
| Phoneme transcription | `vitouphy/wav2vec2-xls-r-300m-phoneme` | ~315 M | local |
| Coach LLM (Tiny Titan mode) | `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` (Q4_K_M) | 4 B | local (llama.cpp) |
| Coach LLM (hosted Space) | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | 8 B | HF Inference router |
| Voice (TTS) | `pocket-tts` | ~250 M | local |

> ⚠️ **About the hosted Space right now:** the public Space is using the **HF Inference router (8 B Nemotron)** for coach replies because the CPU-tier Space can't run the 4 B GGUF in-process fast enough (~5 min per turn, sometimes broken). Flip `COACH_BACKEND=local` to use the **4 B local model**, which is what qualifies the project for the Tiny Titan badge. See "Coach backend" below.

Largest model in the local Tiny Titan pipeline: 4 B parameters. Both modes stay well under the Build Small 32 B cap.

## Features

- Five curriculum levels: Syllables, Starting Words, Middle/End, Phrases, Twisters
- Daily tongue twister that rotates by date (everyone gets the same one each day)
- Streaks, XP, level-ups, with auto-advance after 5 in a row
- Custom UI with no default Gradio chrome: sidebar nav, level path, blob coach character, profile view
- HF OAuth and persistent progress via a private HF Dataset (`IndianChess/rivet-progress`)
- Local Nemotron coach running through llama.cpp

## Run locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# espeak-ng is a system dep for phonemizer
brew install espeak-ng              # macOS
# sudo apt-get install espeak-ng    # Debian/Ubuntu

cp .env.example .env                # optional: add HF_TOKEN for OAuth progress sync
.venv/bin/python app.py             # opens on http://127.0.0.1:7860
```

> ⏳ **First boot downloads ~3 GB.** wav2vec2 (~315 MB), Nemotron GGUF (~2.84 GB), and pocket-tts (~250 MB) get cached into `~/.cache/huggingface/hub/`. Every boot after the first is instant.

## Coach backend

Two modes, picked by the `COACH_BACKEND` env var:

**`router` (default, used by the hosted Space)**
Calls `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` through the HF Inference router. Needs `HF_TOKEN` set (Space secret, or `.env` locally). Fast and reliable, but pulls inference off-device, so this mode does *not* qualify for the Tiny Titan badge.

**`local` (Tiny Titan mode)**
Loads `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` in-process through llama.cpp. The Q4_K_M quant is 2.84 GB on disk and runs on Metal, CUDA, or CPU depending on the host. On Apple Silicon, warm coach turns come back in a couple of seconds. The first turn after boot is slower because of prefill. This is the mode that fits the 4 B Tiny Titan brief.

```bash
# Run with the local 4B GGUF (Tiny Titan)
COACH_BACKEND=local python app.py

# Pick a different GGUF / quant
COACH_BACKEND=local \
COACH_LOCAL_REPO=lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF \
COACH_LOCAL_QUANT=Q6_K \
python app.py
```

To flip the hosted Space to Tiny Titan mode, set `COACH_BACKEND=local` as a Space variable and redeploy on a hardware tier that can carry llama.cpp (ZeroGPU or a paid CPU upgrade).

## Deploy on Hugging Face Spaces

1. Create a new Space inside the **Build Small hackathon org** with `SDK = Gradio`.
2. Push this repo to the Space's git remote.
3. Add `HF_TOKEN` as a Space secret (Settings → Variables and secrets):
   - Grab a read token at <https://huggingface.co/settings/tokens>.
   - It's required for OAuth progress sync. The app still boots without it, but progress goes session-only.

## Building your own word list

`words.py` ships with dictionary IPA. If you want sharper phoneme matching, drop your own reference recordings into `recordings/<word>.wav` and run:

```bash
python build_word_list.py
```

It transcribes each recording with the same wav2vec2 model the app uses, then lets you confirm or edit the IPA before saving.

---

## Field notes

Things people have asked while I was building it.

**Does it work offline?**
Yes. Nemotron runs in-process via llama.cpp, and wav2vec2 plus pocket-tts are on-device, so the whole practice loop is local.

**Why /r/?**
American /r/ is the most-mispronounced consonant in English, and kids who can't produce it get pulled into speech therapy more than for any other phoneme. Most home tools just show flashcards. No acoustic feedback, no progression.

**Is the scoring real, or just vibes?**
Real. wav2vec2 produces an IPA phoneme string for the recording, Praat (via `parselmouth`) extracts the third formant (F3) from the vowel, and the scorer checks whether F3 dipped low enough. A low F3 is the acoustic signature of a retroflex /r/, and that's what catches the classic "wabbit-for-rabbit" substitution that flashcard apps miss entirely.

**Why Nemotron-3-Nano-4B for the coach?**
Small enough to fit on a free CPU Space (2.84 GB at Q4_K_M), fast enough on Apple Silicon to feel live, and good enough at instruction-following that the coach gives you something specific ("curl your tongue back further") instead of "good try!". For this UX, that's the right size.

**How small is the total footprint?**
~3.4 GB on disk. The biggest model is the 4 B Nemotron coach; the other two are in the hundreds of millions. Well under the hackathon's 32 B cap, aimed at Tiny Titan.

**Why a custom UI instead of default Gradio?**
Default Gradio reads like a developer tool. For something kids and adults are going to open every day, the UI had to feel like a product: sidebar nav, a level path with locked checkpoints, a blob character that reacts, a profile with streak and XP, a daily Twister tab. Every visible surface was rebuilt in CSS and Gradio HTML primitives.

**Can an SLP use this clinically?**
No. It's a practice tool, not a clinical instrument. Use it as homework between sessions, not as a replacement for one.

---

## Disclaimer

This is a practice tool, not a substitute for a licensed speech-language pathologist. Recordings are processed locally for feedback generation.

---

Built for the **Hugging Face × Gradio Build Small Hackathon**, June 2026. Backyard AI track. Targeting the NVIDIA Nemotron Hardware Prize and the OpenAI Codex prize.
