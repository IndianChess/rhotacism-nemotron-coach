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
---

# Rivet R Coach

A Gradio app that helps people practice the English /r/ sound. It transcribes
your recording into IPA phonemes using a wav2vec2 phoneme model, measures the
F3 formant with Praat, and a Nemotron LLM gives warm, specific coaching feedback.

## Repository

Public GitHub repo: https://github.com/IndianChess/rhotacism-nemotron-coach

## Deploying on Hugging Face Spaces

1. Create a new Space with **SDK = Gradio**.
2. Push this repo to the Space's remote (instructions below).
3. **Set your `HF_TOKEN` as a Space secret**:
   - Go to your Space → **Settings** → **Variables and secrets**
   - Click **New secret**
   - Name: `HF_TOKEN`
   - Value: a read token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Save and let the Space rebuild.

The `HF_TOKEN` is required for the LLM coach (Nemotron inference). If it's missing,
the app will still launch and score recordings, but coach feedback will fall
back to a generic message.

> ⏳ **First request may take a minute** while the wav2vec2 phoneme model
> (~315 MB) is downloaded and warmed up.

## Local development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# espeak-ng is a system dep for phonemizer:
brew install espeak-ng        # macOS
# sudo apt-get install espeak-ng   # Debian/Ubuntu

cp .env.example .env          # then add your HF_TOKEN
python app.py
```

### Building your own word list

The included `words.py` is a stub using dictionary IPA. For better phoneme
matching, run `build_word_list.py` with your own recordings placed in
`recordings/<word>.wav` — it will transcribe each recording with the same
model the app uses and let you confirm or edit the IPA before saving.

```bash
python build_word_list.py
```

## Disclaimer

This is a practice tool, not a substitute for a licensed speech-language
pathologist. Recordings are processed locally; transcripts are sent to a
hosted LLM (Nemotron) for feedback generation.

Configuration reference: https://huggingface.co/docs/hub/spaces-config-reference
