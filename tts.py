"""Kyutai pocket-tts streaming TTS wrapper.

API verified against pocket-tts==2.1.0:
    model = TTSModel.load_model()                       # ~6s, ~150MB
    state = model.get_state_for_audio_prompt("alba")    # voice prompt
    for chunk in model.generate_audio_stream(state, text):
        chunk: torch.float32 1D tensor, 1920 samples at 24kHz (80ms)

Voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma.
"alba" is a warm female English voice.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterator

import numpy as np

DEFAULT_VOICE = os.environ.get("POCKET_TTS_VOICE", "alba")


@lru_cache(maxsize=1)
def _load_engine():
    from pocket_tts import TTSModel
    print("[tts] Loading pocket-tts...")
    model = TTSModel.load_model()
    print(f"[tts] Loaded. sample_rate={model.sample_rate}Hz")
    return model


@lru_cache(maxsize=10)
def _load_state(voice_name: str):
    return _load_engine().get_state_for_audio_prompt(voice_name)


def sample_rate() -> int:
    return _load_engine().sample_rate


def _chunk_to_int16(chunk) -> np.ndarray:
    audio = chunk.detach().cpu().numpy().astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def synthesize_full(text: str, voice: str = DEFAULT_VOICE) -> tuple[int, np.ndarray]:
    """One-shot synthesis. Returns (sample_rate, int16 mono audio)."""
    model = _load_engine()
    state = _load_state(voice)
    chunks = []
    for chunk in model.generate_audio_stream(state, text):
        chunks.append(_chunk_to_int16(chunk))
    audio = (
        np.concatenate(chunks)
        if chunks
        else np.zeros(0, dtype=np.int16)
    )
    return model.sample_rate, audio


def speak_stream(text: str, voice: str = DEFAULT_VOICE) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (sample_rate, int16 mono) chunks as they decode (~80ms each)."""
    model = _load_engine()
    state = _load_state(voice)
    sr = model.sample_rate
    for chunk in model.generate_audio_stream(state, text):
        yield (sr, _chunk_to_int16(chunk))


if __name__ == "__main__":
    import time
    text = "Hi! I am Wren. Let us try the word 'red' together."
    t0 = time.time()
    for i, (sr, chunk) in enumerate(speak_stream(text)):
        if i == 0:
            print(f"first chunk after {time.time()-t0:.2f}s, sr={sr}, samples={len(chunk)}")
    print(f"total {time.time()-t0:.2f}s, {i+1} chunks")
