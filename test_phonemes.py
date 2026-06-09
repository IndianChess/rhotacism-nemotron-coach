import argparse
import os
import sys

import librosa
import numpy as np
import parselmouth
import torch
from transformers import AutoProcessor, AutoModelForCTC

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser(description="Transcribe IPA phonemes and measure F3 from a .wav file.")
    parser.add_argument("wav_path", help="Path to a .wav audio file")
    args = parser.parse_args()

    if not os.path.isfile(args.wav_path):
        print(f"ERROR: file not found: {args.wav_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Loading audio from {args.wav_path} at {SAMPLE_RATE} Hz...")
    audio, sr = librosa.load(args.wav_path, sr=SAMPLE_RATE)
    duration = len(audio) / sr
    print(f"      Loaded {len(audio)} samples ({duration:.2f}s)")

    print(f"[2/4] Loading model {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID)
    model.eval()
    print("      Model loaded.")

    print("[3/4] Running phoneme recognition...")
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(f"      Phonemes: {transcription}")

    print("[4/4] Computing F3 formant with parselmouth...")
    sound = parselmouth.Sound(args.wav_path)
    formants = sound.to_formant_burg()
    times = np.linspace(0, sound.duration, num=100)
    f3_values = [formants.get_value_at_time(3, t) for t in times]
    f3_values = [v for v in f3_values if v is not None and not np.isnan(v)]
    if f3_values:
        f3_mean = float(np.mean(f3_values))
        print(f"      Mean F3: {f3_mean:.1f} Hz (from {len(f3_values)} frames)")
    else:
        print("      No F3 values detected.")


if __name__ == "__main__":
    main()
