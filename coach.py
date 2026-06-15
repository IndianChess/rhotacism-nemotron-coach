"""Wren, the /r/ practice coach.

Two interchangeable backends:
  * router  (default) — HF Inference Providers, Llama-3.1-Nemotron-Nano-8B-v1
              served by featherless-ai. Cloud, fast, needs HF_TOKEN.
  * local             — llama-cpp-python loading NVIDIA-Nemotron-3-Nano-4B
              (Q4_K_M GGUF, ~2.84 GB). Zero cloud calls, qualifies for the
              hackathon's "Off the Grid" bonus badge.

Pick via COACH_BACKEND env var. coach_turn() signature is identical for
both — callers don't care which path runs.

coach_turn(state, score_dict, user_transcript) -> dict
    spoken_reply       (str, < 55 words for low TTS latency)
    next_target_word   (str | None)
    cue_type           ("retroflex" | "bunched" | "shaping_from_ear" |
                        "auditory_discrimination" | "needs_lowering" | "none")
    is_correct         (bool)
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# Zero GPU detection. spaces.GPU is a no-op decorator when not on Zero GPU
# hardware, but we still want to know whether we're running there so we can
# skip the eager startup preload (the GPU isn't allocated at import time).
ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
try:
    import spaces  # type: ignore
    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False
    class _SpacesShim:
        @staticmethod
        def GPU(*_a, **_kw):
            def deco(fn):
                return fn
            return deco
    spaces = _SpacesShim()  # type: ignore

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
# Default to the HF Inference router because the in-process llama.cpp path
# is too slow on CPU-tier Spaces and unreliable on ZeroGPU. To run the model
# locally (e.g. Apple Silicon dev with Metal), set COACH_BACKEND=local.
BACKEND = os.environ.get("COACH_BACKEND", "router").lower()
if BACKEND not in {"router", "local"}:
    print(f"[coach] unknown COACH_BACKEND={BACKEND!r}, falling back to 'router'")
    BACKEND = "router"

# --- router backend (HF Inference Providers) ---
ROUTER_MODEL_ID = os.environ.get(
    "COACH_MODEL_ID",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1:featherless-ai",
)
ROUTER_BASE_URL = "https://router.huggingface.co/v1"

# Backwards-compat alias (used by anything that imported MODEL_ID directly).
MODEL_ID = ROUTER_MODEL_ID
BASE_URL = ROUTER_BASE_URL

# --- local backend (llama.cpp + GGUF) ---
LOCAL_REPO  = os.environ.get(
    "COACH_LOCAL_REPO",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
)
LOCAL_QUANT = os.environ.get("COACH_LOCAL_QUANT", "Q4_K_M")
# The coach SYSTEM_PROMPT below is ~3.5k tokens; combined with the user
# message + room for the JSON reply we routinely cross 4096. 8192 gives
# comfortable headroom without bloating the KV cache. Nemotron-3-Nano-4B
# itself supports up to 262K, so this is the safe lower bound, not a ceiling.
LOCAL_CTX   = int(os.environ.get("COACH_LOCAL_CTX", "8192"))

SYSTEM_PROMPT = """You are Wren, a direct and encouraging speech coach helping an 18-year-old fix their /r/ sound. Your words will be spoken aloud by a text-to-speech voice — write for the ear, not the eye. Be concise, specific, and treat the user as a competent adult.

═══════════════════════════════════════════════════════════════════════
VOCABULARY RULES — strict
═══════════════════════════════════════════════════════════════════════

NEVER say: formant, F3, hertz, Hz, phoneme, IPA, retroflex, bunched,
  rhotic, alveolar, articulation, score, percent, accuracy, or any
  number from the scoring data (no "your F3 was 2400", no "0.6 score").

USE these anatomical terms freely:
  tongue, tongue tip, tongue back/root, lips, teeth, jaw, roof of mouth,
  the ridge behind your top teeth (or "the bump"), throat, mirror.

USE these verbs: curl, bunch, pull back, push up, hold, freeze, drop,
  spread, round, open, relax, point, press, lift.

═══════════════════════════════════════════════════════════════════════
EXERCISE TYPES — tone shifts by context
═══════════════════════════════════════════════════════════════════════

SYLLABLE mode (ra, re, ri, etc.):
  The user is drilling the isolated /r/ sound. Focus only on tongue
  position. Keep intro very short: name the syllable, give ONE tip, go.

WORD mode (single words):
  Standard practice. Use the 3-part structure below.

PHRASE mode (two or more words):
  Harder — /r/ in connected speech. Acknowledge the added difficulty.
  Focus on keeping tongue position stable across the whole phrase.

═══════════════════════════════════════════════════════════════════════
CRITICAL: WHEN r_quality IS "correct" — CELEBRATE, DON'T CORRECT
═══════════════════════════════════════════════════════════════════════

If the scoring data says r_quality is "correct", the user produced a
clean /r/. They do NOT need a corrective cue. They need acknowledgment.

Format: ONE specific celebration sentence (≤12 words) + one short ask
to repeat or move on. Total under 25 words. Set is_correct = true.

  GOOD: "Clean R — tongue stayed in position the whole time. Nice."
  GOOD: "That's the one — locked in. Try it once more."
  BAD:  "Good try, but make sure to curl your tongue..." (DON'T CORRECT)
  BAD:  "Almost — try again with your tongue tip up." (DON'T CORRECT)

If r_quality is "approaching" — acknowledge the progress but offer the
needs_lowering cue. Set is_correct = false but be warm.

═══════════════════════════════════════════════════════════════════════
FEEDBACK STRUCTURE — every imperfect attempt (under 55 words total)
═══════════════════════════════════════════════════════════════════════

  1) SHORT REACTION (≤8 words) — be specific, not generic.
       GOOD: "That R softened near the end."
       GOOD: "Your lips rounded a bit — keep them flat."
       BAD:  "Good try!" / "Keep it up!"

  2) ONE PHYSICAL CUE from the cue bank (1–2 sentences).
     One cue only — never stack two.

  3) CLEAR NEXT INSTRUCTION (1 sentence).
       GOOD: "Try it again, slow."
       GOOD: "Stretch the R: 'rrrrred'."

═══════════════════════════════════════════════════════════════════════
CUE BANK — pick exactly ONE per reply based on what went wrong
═══════════════════════════════════════════════════════════════════════

→ W-SUBSTITUTION (error_detail = "w_substitution"):
  "Check your lips — are they rounding into a circle? Spread them flat.
  Only your tongue moves for R, not your lips."

→ APPROACHING / F3 BORDERLINE (error_detail = "needs_lowering" or
  r_quality = "approaching"):
  "Your tongue is almost there. Pull the back of your tongue a little
  further back and down — like you're making a bit more space in your
  throat. Hold that shape."

→ FLAT TONGUE / DISTORTION (r_quality = "unclear", error_detail = "distortion"):
  "Curl your tongue tip up toward the bump just behind your top front
  teeth. Don't let it touch — just aim at it and hold."

→ ALTERNATIVE SHAPE — bunched (use after 2+ retroflex cues fail):
  "Try a different shape: push the BACK of your tongue up toward the
  roof of your mouth, like hiding something behind your back teeth.
  Tongue tip points down."

→ ALTERNATIVE SHAPE — retroflex (use after 2+ bunched cues fail):
  "Switch it up: curl the tongue tip up and back, pointing toward the
  bump behind your top teeth. Keep it there the whole time you say R."

→ TONGUE DROPS MID-WORD (R starts okay but fades):
  "Hold the R shape a half-second longer before moving on — like
  'rrrrred'. Don't let your tongue escape early."

→ SHAPING FROM A SOUND YOU ALREADY KNOW (use shaping_from_ear when
  the user is stuck on isolated R but can produce 'ear' or 'er'):
  "Say 'ear' slowly — feel where your tongue ends up at the very end?
  That's the R position. Now start from there: 'ear...r-ed'."

→ TIGHT JAW:
  "Open your jaw a bit more — a finger's width between your teeth.
  A clenched jaw blocks the R."

→ OMISSION (error_detail = "omission"):
  "The R got lost this time. Start from the R before anything else —
  get your tongue in position first, then say the word."

→ AFTER 3+ FAILED ATTEMPTS — ear training:
  "Let's reset. I'll say the word twice — notice exactly where the R
  sound sits. Then try once more."

═══════════════════════════════════════════════════════════════════════
WORD SELECTION
═══════════════════════════════════════════════════════════════════════

Always work with the EXACT "Current target" in the user message.
Set next_target_word equal to it. Never suggest moving to a different
target — the app controls progression.

On a CORRECT attempt: celebrate briefly (specific, not generic), then
ask them to repeat once to lock it in. Don't move them on — the Next
button does that.
  GOOD: "That's it — clean R, tongue held the shape. Do it once more."
  BAD:  "Great! Now let's try 'tree'."

═══════════════════════════════════════════════════════════════════════
ADAPT TO HISTORY
═══════════════════════════════════════════════════════════════════════

- Same word failed 3+ times → switch cue type AND try the ear-training
  framing. Stay on the same word.
- Correct 2+ times in a row → get briefer. "Locked in. One more."
- After a struggle then a win → be specific about what changed:
  "That time your tongue stayed in position — that's the difference."

═══════════════════════════════════════════════════════════════════════
INTRO MODE (no scoring data, context = INTRODUCING)
═══════════════════════════════════════════════════════════════════════

Structure: (a) say the target clearly, (b) ONE physical tip, (c) invite.
Under 40 words. Name the exact word given — never substitute another.

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — JSON only, no markdown, no prose around it
═══════════════════════════════════════════════════════════════════════

{
  "spoken_reply":     string  (under 55 words, ear-friendly, no banned vocab),
  "next_target_word": string or null,
  "cue_type":         "retroflex"|"bunched"|"shaping_from_ear"|
                      "auditory_discrimination"|"needs_lowering"|"none",
  "is_correct":       boolean
}

═══════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════

Ex 1 — Word "red", error_detail=w_substitution, first attempt:
{
  "spoken_reply": "Lips rounded — that came out as a W. Spread your lips flat and keep them still. Only your tongue does the work for R. Try again.",
  "next_target_word": "red",
  "cue_type": "retroflex",
  "is_correct": false
}

Ex 2 — Syllable "ra", r_quality=approaching:
{
  "spoken_reply": "Getting close. Pull the back of your tongue a little further back — make a bit more space in your throat. Now say 'ra' again.",
  "next_target_word": "ra",
  "cue_type": "needs_lowering",
  "is_correct": false
}

Ex 3 — Word "rocket", r_quality=correct, second correct in a row:
{
  "spoken_reply": "There it is — R held the whole way through. Once more to lock it in.",
  "next_target_word": "rocket",
  "cue_type": "none",
  "is_correct": true
}

Ex 4 — Intro for phrase "red rose":
{
  "spoken_reply": "Now say 'red rose' — keep your tongue in R position as you move from the first word into the second. Don't drop it between words. Go.",
  "next_target_word": "red rose",
  "cue_type": "retroflex",
  "is_correct": false
}
"""


@lru_cache(maxsize=1)
def _get_router_client():
    """OpenAI-protocol client pointed at HF's Inference Providers router."""
    from openai import OpenAI
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set — cannot call the router backend. "
            "Either set HF_TOKEN or switch to COACH_BACKEND=local."
        )
    return OpenAI(base_url=ROUTER_BASE_URL, api_key=token)


@lru_cache(maxsize=1)
def _get_local_llm():
    """Lazily build the llama-cpp-python handle and cache it for the process."""
    from llama_cpp import Llama
    print(f"[coach] loading local model {LOCAL_REPO} ({LOCAL_QUANT})...")
    t0 = time.time()
    llm = Llama.from_pretrained(
        repo_id=LOCAL_REPO,
        filename=f"*{LOCAL_QUANT}*",
        n_ctx=LOCAL_CTX,
        n_gpu_layers=-1,   # full Metal/CUDA offload where available; harmless on CPU
        n_batch=1024,      # bigger prefill batches → faster prompt processing on Metal
        n_threads=int(os.environ.get("COACH_LOCAL_THREADS", "0")) or None,
        verbose=os.environ.get("COACH_LOCAL_VERBOSE", "0") == "1",
    )
    print(f"[coach] local model loaded in {time.time() - t0:.1f}s")
    return llm


def _chat_router(messages: list[dict]) -> str:
    """One JSON-mode round trip to the HF router. Returns raw assistant text."""
    client = _get_router_client()
    resp = client.chat.completions.create(
        model=ROUTER_MODEL_ID,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.6,
        max_tokens=400,
    )
    return resp.choices[0].message.content


def _chat_local(messages: list[dict]) -> str:
    """One JSON-mode round trip to the locally-loaded GGUF model."""
    llm = _get_local_llm()
    resp = llm.create_chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.6,
        max_tokens=400,
    )
    return resp["choices"][0]["message"]["content"]


# Public helper so app.py can preload the model at startup when BACKEND=local
def preload() -> None:
    """Force-load whichever backend we're configured for. Safe to call twice."""
    if BACKEND == "local":
        _get_local_llm()
    else:
        # Router path doesn't really preload — but probe the token now so we
        # fail loudly at startup instead of silently on the first turn.
        try:
            _get_router_client()
        except RuntimeError as e:
            print(f"[coach] {e}")


def _format_recent(history: list[dict]) -> str:
    if not history:
        return "  (no prior attempts)"
    lines = []
    for h in history[-5:]:
        ex_type = h.get("exercise_type", "word")
        lines.append(
            f"  - [{ex_type}] target={h.get('target_word')!r}, "
            f"r_quality={h.get('r_quality')}, "
            f"error_detail={h.get('error_detail')!r}, "
            f"score={h.get('overall_score')}, "
            f"correct={h.get('is_correct')}"
        )
    return "\n".join(lines)


def _build_user_message(
    state: dict,
    score_dict: dict | None,
    user_transcript: str | None,
) -> str:
    target = state.get("current_target_word") or "(none)"
    history = state.get("history", [])
    exercise_type = state.get("exercise_type", "word")

    type_label = {"syllable": "SYLLABLE", "phrase": "PHRASE"}.get(exercise_type, "WORD")

    if score_dict is None:
        if not history:
            context = (
                f"FIRST TURN — introduce yourself briefly, then introduce the "
                f"{type_label} target '{target}' with one physical tip. "
                f"Your spoken_reply MUST contain '{target}'."
            )
        else:
            context = (
                f"INTRODUCING NEW {type_label} TARGET — say '{target}' and give "
                f"one physical tip for producing it. Your spoken_reply MUST "
                f"contain '{target}'."
            )
    else:
        context = (
            f"GIVING FEEDBACK on attempt at {type_label} '{target}'. "
            f"Translate scoring into concrete body cues — no numbers or jargon."
        )

    parts = [
        f"Context: {context}",
        f"Exercise type: {type_label}",
        f"Current target: {target}",
        f"User said (STT): {user_transcript!r}" if user_transcript else "User said: (no transcript)",
    ]

    if score_dict:
        parts += [
            "Scoring data:",
            f"  detected_phonemes: {score_dict.get('detected_phonemes')!r}",
            f"  target_phonemes:   {score_dict.get('target_phonemes')!r}",
            f"  r_quality:         {score_dict.get('r_quality')}",
            f"  error_detail:      {score_dict.get('error_detail')}",
            f"  f3_hz:             {score_dict.get('f3_hz')} Hz "
              "(< 2400 = good /r/; > 2600 = /w/-like)",
            f"  phoneme_match:     {score_dict.get('phoneme_match')}",
            f"  overall_score:     {score_dict.get('overall_score')}",
        ]
    else:
        parts.append("Scoring: (none — intro turn)")

    parts.append("Recent attempts:")
    parts.append(_format_recent(history))
    parts.append("\nOutput JSON only.")
    return "\n".join(parts)


_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()

    # Strip model reasoning block if present
    if "<think>" in s:
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

    m = _JSON_FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Truncated JSON recovery
    if s.startswith("{"):
        for trim in range(len(s), 0, -1):
            candidate = s[:trim].rstrip().rstrip(",")
            opens = candidate.count("{") - candidate.count("}")
            if opens > 0:
                candidate += "}" * opens
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


_VALID_CUES = {
    "retroflex", "bunched", "shaping_from_ear",
    "auditory_discrimination", "needs_lowering", "none",
}


def _default_response(state: dict, score_dict: dict | None) -> dict:
    target = state.get("current_target_word") or "red"
    ex_type = state.get("exercise_type", "word")

    if not score_dict:
        if ex_type == "syllable":
            return {
                "spoken_reply": (
                    f"Let's work on '{target}'. Curl your tongue tip up toward the bump "
                    f"behind your top teeth and hold it. Say '{target}'."
                ),
                "next_target_word": target,
                "cue_type": "retroflex",
                "is_correct": False,
            }
        return {
            "spoken_reply": (
                f"Let's try '{target}'. Curl your tongue tip up toward the bump behind "
                f"your top front teeth. Keep your lips relaxed. Go."
            ),
            "next_target_word": target,
            "cue_type": "retroflex",
            "is_correct": False,
        }

    r_q = score_dict.get("r_quality")
    err = score_dict.get("error_detail", "")

    if r_q == "correct":
        return {
            "spoken_reply": "Clean R — tongue held the shape. Do it once more to lock it in.",
            "next_target_word": target,
            "cue_type": "none",
            "is_correct": True,
        }
    if err == "w_substitution":
        return {
            "spoken_reply": (
                "Lips rounded there — that's a W. Spread your lips flat and keep them still. "
                "Only your tongue works for R. Try again."
            ),
            "next_target_word": target,
            "cue_type": "retroflex",
            "is_correct": False,
        }
    if err == "needs_lowering":
        return {
            "spoken_reply": (
                "Getting there. Pull the back of your tongue a bit further back and down — "
                "create more space in your throat. Hold that and try again."
            ),
            "next_target_word": target,
            "cue_type": "needs_lowering",
            "is_correct": False,
        }
    return {
        "spoken_reply": (
            "Curl your tongue tip up toward the bump behind your top teeth. "
            "Keep it there the whole time you say the R. Try again."
        ),
        "next_target_word": target,
        "cue_type": "retroflex",
        "is_correct": False,
    }


def _validate(parsed: dict, default: dict) -> dict:
    out = dict(default)
    if isinstance(parsed.get("spoken_reply"), str) and parsed["spoken_reply"].strip():
        out["spoken_reply"] = parsed["spoken_reply"].strip()
    if "next_target_word" in parsed:
        v = parsed["next_target_word"]
        if v is None or (isinstance(v, str) and v.strip()):
            out["next_target_word"] = v.strip() if isinstance(v, str) else None
    if isinstance(parsed.get("cue_type"), str) and parsed["cue_type"] in _VALID_CUES:
        out["cue_type"] = parsed["cue_type"]
    if isinstance(parsed.get("is_correct"), bool):
        out["is_correct"] = parsed["is_correct"]
    return out


def coach_turn(
    state: dict,
    score_dict: dict | None,
    user_transcript: str | None,
) -> dict:
    default = _default_response(state, score_dict)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_message(state, score_dict, user_transcript)},
    ]

    t0 = time.time()
    try:
        content = _chat_local(messages) if BACKEND == "local" else _chat_router(messages)
    except Exception as e:
        print(f"[coach] {BACKEND} inference error: {e}")
        return default
    label = LOCAL_REPO if BACKEND == "local" else ROUTER_MODEL_ID
    print(f"[coach] {BACKEND}:{label} round-trip {time.time() - t0:.2f}s")

    parsed = _parse_json(content)
    if not isinstance(parsed, dict):
        print(f"[coach] unparseable: {content!r}")
        return default
    return _validate(parsed, default)


# Backward-compatible wrapper
def generate_feedback(scoring_result: dict, target_word: str, attempt_history: list) -> dict:
    state = {"current_target_word": target_word, "history": attempt_history}
    result = coach_turn(state, scoring_result, None)
    return {
        "verdict":      "That R landed!" if result["is_correct"] else "Let's try again.",
        "feedback":     result["spoken_reply"],
        "encouragement":"Locked in." if result["is_correct"] else "You're making progress.",
        "suggest_next": "harder" if result["is_correct"] else "same",
    }


if __name__ == "__main__":
    state = {"current_target_word": "red", "history": [], "exercise_type": "word"}
    print("--- intro ---")
    print(coach_turn(state, None, None))

    score = {
        "detected_phonemes": "w ɛ d",
        "target_phonemes":   "r ɛ d",
        "phoneme_match":     0.66,
        "f3_hz":             2700.0,
        "r_quality":         "substituted_w",
        "error_detail":      "w_substitution",
        "overall_score":     0.4,
    }
    print("\n--- w-sub feedback ---")
    print(coach_turn(state, score, "wed"))
