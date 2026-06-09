"""Phoneme scoring for /r/ production.

Key improvements over v1:
  • CTC per-frame probabilities are used to find the actual /r/ segment
    before computing F3 — no more "lowest 15% of the whole word" proxy.
  • F3 threshold lowered to 2400 Hz (calibrated for young adults).
  • _classify_r returns 5 classes: correct | approaching | unclear |
    substituted_w | omitted.
  • score_pronunciation now returns error_detail and r_segment times.
"""

import os

import librosa
import numpy as np
import parselmouth
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForCTC

load_dotenv()

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16000
# F3 thresholds calibrated for young adult speakers (18–25). For females
# and brighter male voices a clean /r/ can sit up to ~2500 Hz, so anything
# below 2550 reads as correct. Above 2750 is definitively /w/-like.
F3_R_THRESHOLD_HZ = 2550.0
F3_APPROACHING_HZ = 2750.0
FRAME_STRIDE_S = 0.02        # wav2vec2 feature stride: 320 samples @ 16kHz

_processor = None
_model = None


def _get_model():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForCTC.from_pretrained(MODEL_ID)
        _model.eval()
    return _processor, _model


# ---------------------------------------------------------------------------
# Phoneme utilities
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    a_toks = a.split()
    b_toks = b.split()
    n, m = len(a_toks), len(b_toks)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if a_toks[i - 1] == b_toks[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def _normalize_phonemes(s: str) -> str:
    """
    Collapse phoneme variants so target ↔ detected comparison is fair.

    The espeak phoneme model often emits a different IPA symbol for the
    same English vowel sound than the one a human typically writes (e.g.
    'a' vs 'æ' for the vowel in "rabbit", 'i' vs 'ɪ' for the vowel in
    "rip"). Penalising those cosmetic differences in the score tanks the
    overall number for users who actually pronounced the word correctly.
    """
    # /r/ family
    s = s.replace("ɹ", "r")
    s = s.replace("ɝː", "ɜ r")
    s = s.replace("ɝ", "ɜ r")
    s = s.replace("ɚ", "ə r")

    # Vowel variant collapsing — the espeak model and our hand-coded IPA
    # use different symbols for the same vowel sound.
    s = s.replace("ɪ", "i")
    s = s.replace("æ", "a")
    s = s.replace("ɛ", "e")
    s = s.replace("ɔ", "o")
    s = s.replace("ɑ", "a")
    s = s.replace("ʊ", "u")
    s = s.replace("ʌ", "ə")

    # Length marker doesn't affect phoneme identity.
    s = s.replace("ː", "")

    # American flap-t (as in "butter") often emitted as ɾ.
    s = s.replace("ɾ", "t")

    return s.strip()


def _phoneme_similarity(detected: str, target: str) -> float:
    detected = _normalize_phonemes(detected)
    target = _normalize_phonemes(target)
    if not detected and not target:
        return 1.0
    dist = _levenshtein(detected, target)
    max_len = max(len(detected.split()), len(target.split()))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - dist / max_len)


# ---------------------------------------------------------------------------
# CTC timestamp extraction for /r/ segment
# ---------------------------------------------------------------------------

def _extract_r_frame_range(
    logits: torch.Tensor,
    processor,
) -> tuple[float | None, float | None]:
    """
    Find the time window in the audio where /r/-class phonemes appear,
    using per-frame softmax probabilities from wav2vec2 CTC logits.

    Returns (start_sec, end_sec) or (None, None) if no /r/ detected.
    """
    vocab = processor.tokenizer.get_vocab()
    # /r/ family tokens emitted by the espeak phoneme model
    r_tokens = {"r", "ɹ", "ɝ", "ɚ", "ɝː"}
    r_ids = [idx for token, idx in vocab.items() if token in r_tokens]

    if not r_ids:
        return None, None

    probs = torch.softmax(logits[0], dim=-1)          # (T, vocab)
    r_ids_t = torch.tensor(r_ids, dtype=torch.long)
    r_prob = probs[:, r_ids_t].sum(dim=-1)            # (T,)

    # Primary: probability threshold
    THRESH = 0.08
    r_frames = torch.where(r_prob > THRESH)[0].tolist()

    # Fallback: argmax (CTC blank dominates but /r/ may still be argmax briefly)
    if not r_frames:
        preds = torch.argmax(logits[0], dim=-1).tolist()
        r_ids_set = set(r_ids)
        r_frames = [i for i, p in enumerate(preds) if p in r_ids_set]

    if not r_frames:
        return None, None

    n = logits.shape[1]
    buf = 2  # 40ms buffer either side
    t_start = max(0, r_frames[0] - buf) * FRAME_STRIDE_S
    t_end = (min(n - 1, r_frames[-1] + buf) + 1) * FRAME_STRIDE_S
    return t_start, t_end


# ---------------------------------------------------------------------------
# F3 formant measurement
# ---------------------------------------------------------------------------

def _estimate_r_f3(
    audio_path: str,
    r_start: float | None = None,
    r_end: float | None = None,
) -> float | None:
    """
    Measure the third formant (F3).

    When r_start/r_end are provided (from CTC timestamps), measure F3 within
    the /r/ segment specifically.  Falls back to the lowest-15% heuristic.
    """
    sound = parselmouth.Sound(audio_path)
    formants = sound.to_formant_burg()
    duration = sound.duration

    if r_start is not None and r_end is not None:
        t0 = max(0.0, r_start)
        t1 = min(duration, r_end)
        if t1 - t0 >= 0.02:           # need at least 20ms of /r/ segment
            times = np.linspace(t0, t1, num=40)
            f3 = np.array([formants.get_value_at_time(3, t) for t in times], dtype=float)
            f3 = f3[~np.isnan(f3)]
            if f3.size >= 5:
                return float(np.median(f3))

    # Fallback: whole recording, lowest 15% of frames as proxy for /r/
    times = np.linspace(0, duration, num=200)
    f3 = np.array([formants.get_value_at_time(3, t) for t in times], dtype=float)
    f3 = f3[~np.isnan(f3)]
    if f3.size == 0:
        return None
    k = max(1, int(0.15 * f3.size))
    return float(np.mean(np.sort(f3)[:k]))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify_r(detected_phonemes: str, f3_hz: float | None) -> str:
    """
    Five-class /r/ quality classifier.

      correct      — phoneme model heard /r/ AND F3 is appropriately low
      approaching  — F3 is borderline; tongue is in the right zone
      unclear      — /r/ detected phonemically but F3 is /w/-like
      substituted_w — /w/ heard instead of /r/
      omitted      — no /r/ or /w/ heard in an /r/ target context
    """
    toks = _normalize_phonemes(detected_phonemes).split()
    has_r = "r" in toks
    has_w = "w" in toks

    # No F3 measurement available — trust the phoneme model.
    # (Used to return "unclear" here, which incorrectly downgraded clean
    # attempts on very short recordings or noisy mics.)
    if f3_hz is None:
        if has_r:
            return "correct"
        if has_w:
            return "substituted_w"
        return "omitted"

    if has_r and f3_hz < F3_R_THRESHOLD_HZ:
        return "correct"
    if has_r and f3_hz < F3_APPROACHING_HZ:
        return "approaching"
    if has_w and not has_r:
        return "substituted_w"
    if has_r:
        return "unclear"    # r detected but F3 is /w/-like
    return "omitted"


def _classify_error(detected: str, target: str, r_quality: str) -> str:
    """
    Map the scorer's output to a single error label Wren can act on.

      none           — no /r/ in target, or the attempt was correct
      w_substitution — /w/ used instead of /r/
      l_substitution — /l/ used instead of /r/
      omission       — /r/ expected but not heard at all
      needs_lowering — tongue shape almost right (F3 borderline)
      distortion     — /r/ detected but acoustic quality off
    """
    d = _normalize_phonemes(detected).split()
    t = _normalize_phonemes(target).split()

    if "r" not in t:
        return "none"
    if r_quality == "correct":
        return "none"
    if r_quality == "approaching":
        return "needs_lowering"
    if r_quality == "substituted_w":
        return "w_substitution"
    if "l" in d and "r" not in d:
        return "l_substitution"
    if "r" not in d:
        return "omission"
    return "distortion"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_pronunciation(
    audio_path: str,
    target_word: str,
    target_phonemes: str,
) -> dict:
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    processor, model = _get_model()
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    detected = processor.batch_decode(predicted_ids)[0].strip()

    # ---- Segment-aware F3 measurement ----
    r_start, r_end = _extract_r_frame_range(logits, processor)
    phoneme_match = _phoneme_similarity(detected, target_phonemes)
    f3_hz = _estimate_r_f3(audio_path, r_start=r_start, r_end=r_end)
    r_quality = _classify_r(detected, f3_hz)
    error_detail = _classify_error(detected, target_phonemes, r_quality)

    # For a /r/-practice app, the /r/ is what matters. The phoneme_match
    # serves as a sanity check that the user said *roughly* the target
    # word — not as a primary quality signal. Heavily weight r_quality.
    r_weight = {
        "correct":      1.0,
        "approaching":  0.85,
        "unclear":      0.5,
        "substituted_w": 0.0,
        "omitted":      0.1,
    }
    overall = 0.25 * phoneme_match + 0.75 * r_weight.get(r_quality, 0.5)

    return {
        "detected_phonemes": detected,
        "target_phonemes":   target_phonemes,
        "phoneme_match":     round(phoneme_match, 3),
        "f3_hz":             round(f3_hz, 1) if f3_hz is not None else None,
        "r_start_s":         round(r_start, 3) if r_start is not None else None,
        "r_end_s":           round(r_end, 3) if r_end is not None else None,
        "r_quality":         r_quality,
        "error_detail":      error_detail,
        "overall_score":     round(overall, 3),
    }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    test_cases = [
        ("good.wav",  "rabbit", "r æ b ɪ t"),
        ("bad.wav",   "rabbit", "r æ b ɪ t"),
        ("other.wav", "red",    "r ɛ d"),
    ]
    for path, word, phonemes in test_cases:
        full = os.path.join(here, path)
        if not os.path.isfile(full):
            continue
        print(f"\n=== {path} (target: {word}) ===")
        result = score_pronunciation(full, word, phonemes)
        for k, v in result.items():
            print(f"  {k}: {v}")
