"""Curated /r/ exercise bank following the SLP clinical hierarchy.

Curriculum levels (maps to `difficulty` field):
  0  Syllable sounds — ra, re, ri, ro, ru, ar, er, or
  1  Initial /r/ words — common, high-frequency
  2  Medial and final /r/ words
  3  Vocalic /r/ and /r/ blends
  4  Short phrases with /r/

The `type` field labels the exercise category:
  "syllable" | "word" | "phrase"

IPA notes:
  • target phonemes always use "r" (not "ɹ"); scoring._normalize_phonemes
    collapses ɹ → r before comparison.
  • Syllable targets are minimal — just the critical phonemes.
"""

import random

EXERCISES = [
    # ── Level 0: Syllable sounds ─────────────────────────────────────────
    # Pre-word practice: isolate the /r/ in CV and VC syllables.
    {"word": "ra",  "phonemes": "r æ",   "position": "syllable", "difficulty": 0, "type": "syllable"},
    {"word": "re",  "phonemes": "r ɛ",   "position": "syllable", "difficulty": 0, "type": "syllable"},
    {"word": "ri",  "phonemes": "r iː",  "position": "syllable", "difficulty": 0, "type": "syllable"},
    {"word": "ro",  "phonemes": "r oʊ",  "position": "syllable", "difficulty": 0, "type": "syllable"},
    {"word": "ru",  "phonemes": "r uː",  "position": "syllable", "difficulty": 0, "type": "syllable"},
    {"word": "ar",  "phonemes": "ɑ r",   "position": "syllable", "difficulty": 0, "type": "syllable"},
    {"word": "er",  "phonemes": "ɜ r",   "position": "syllable", "difficulty": 0, "type": "syllable"},
    {"word": "or",  "phonemes": "ɔ r",   "position": "syllable", "difficulty": 0, "type": "syllable"},
    {"word": "ir",  "phonemes": "ɪ r",   "position": "syllable", "difficulty": 0, "type": "syllable"},
    {"word": "ur",  "phonemes": "ʌ r",   "position": "syllable", "difficulty": 0, "type": "syllable"},

    # ── Level 1: Initial /r/ words ────────────────────────────────────────
    # /r/ is easiest to practice at the start of a word.
    {"word": "red",    "phonemes": "r ɛ d",   "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "run",    "phonemes": "r ʌ n",   "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "rain",   "phonemes": "r eɪ n",  "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "rope",   "phonemes": "r oʊ p",  "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "right",  "phonemes": "r aɪ t",  "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "read",   "phonemes": "r iː d",  "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "real",   "phonemes": "r iː l",  "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "ride",   "phonemes": "r aɪ d",  "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "race",   "phonemes": "r eɪ s",  "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "roof",   "phonemes": "r uː f",  "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "ring",   "phonemes": "r ɪ ŋ",   "position": "initial", "difficulty": 1, "type": "word"},
    {"word": "road",   "phonemes": "r oʊ d",  "position": "initial", "difficulty": 1, "type": "word"},

    # ── Level 2: Medial and final /r/ words ──────────────────────────────
    {"word": "very",   "phonemes": "v ɛ r i",     "position": "medial", "difficulty": 2, "type": "word"},
    {"word": "carrot", "phonemes": "k æ r ə t",   "position": "medial", "difficulty": 2, "type": "word"},
    {"word": "around", "phonemes": "ə r aʊ n d",  "position": "medial", "difficulty": 2, "type": "word"},
    {"word": "story",  "phonemes": "s t ɔ r i",   "position": "medial", "difficulty": 2, "type": "word"},
    {"word": "error",  "phonemes": "ɛ r ə r",     "position": "medial", "difficulty": 2, "type": "word"},
    {"word": "correct","phonemes": "k ə r ɛ k t", "position": "medial", "difficulty": 2, "type": "word"},
    {"word": "car",    "phonemes": "k ɑ r",        "position": "final",  "difficulty": 2, "type": "word"},
    {"word": "four",   "phonemes": "f ɔ r",        "position": "final",  "difficulty": 2, "type": "word"},
    {"word": "star",   "phonemes": "s t ɑ r",      "position": "final",  "difficulty": 2, "type": "word"},
    {"word": "more",   "phonemes": "m ɔ r",        "position": "final",  "difficulty": 2, "type": "word"},
    {"word": "door",   "phonemes": "d ɔ r",        "position": "final",  "difficulty": 2, "type": "word"},
    {"word": "floor",  "phonemes": "f l ɔ r",      "position": "final",  "difficulty": 2, "type": "word"},

    # ── Level 3: Vocalic /r/ and blends ──────────────────────────────────
    # Vocalic /r/ (r-colored vowels) and consonant clusters are the hardest.
    {"word": "bird",   "phonemes": "b ɜ r d",   "position": "vocalic", "difficulty": 3, "type": "word"},
    {"word": "her",    "phonemes": "h ɜ r",      "position": "vocalic", "difficulty": 3, "type": "word"},
    {"word": "butter", "phonemes": "b ʌ t ə r",  "position": "vocalic", "difficulty": 3, "type": "word"},
    {"word": "better", "phonemes": "b ɛ t ə r",  "position": "vocalic", "difficulty": 3, "type": "word"},
    {"word": "teacher","phonemes": "t iː tʃ ə r","position": "vocalic", "difficulty": 3, "type": "word"},
    {"word": "early",  "phonemes": "ɜ r l i",    "position": "vocalic", "difficulty": 3, "type": "word"},
    {"word": "tree",   "phonemes": "t r i",       "position": "blend",   "difficulty": 3, "type": "word"},
    {"word": "dream",  "phonemes": "d r i m",     "position": "blend",   "difficulty": 3, "type": "word"},
    {"word": "frog",   "phonemes": "f r ɑ g",     "position": "blend",   "difficulty": 3, "type": "word"},
    {"word": "green",  "phonemes": "g r i n",     "position": "blend",   "difficulty": 3, "type": "word"},
    {"word": "bread",  "phonemes": "b r ɛ d",     "position": "blend",   "difficulty": 3, "type": "word"},
    {"word": "price",  "phonemes": "p r aɪ s",    "position": "blend",   "difficulty": 3, "type": "word"},

    # ── Level 4: Short phrases with /r/ ──────────────────────────────────
    # Carryover: the hardest setting. Produces /r/ in natural connected speech.
    {"word": "red rose",      "phonemes": "r ɛ d r oʊ z",         "position": "phrase", "difficulty": 4, "type": "phrase"},
    {"word": "run fast",      "phonemes": "r ʌ n f æ s t",         "position": "phrase", "difficulty": 4, "type": "phrase"},
    {"word": "really great",  "phonemes": "r iː l i g r eɪ t",    "position": "phrase", "difficulty": 4, "type": "phrase"},
    {"word": "right here",    "phonemes": "r aɪ t h ɪ r",          "position": "phrase", "difficulty": 4, "type": "phrase"},
    {"word": "green rabbit",  "phonemes": "g r i n r æ b ɪ t",     "position": "phrase", "difficulty": 4, "type": "phrase"},
    {"word": "every year",    "phonemes": "ɛ v r i j ɪ r",         "position": "phrase", "difficulty": 4, "type": "phrase"},
    {"word": "read a book",   "phonemes": "r iː d ə b ʊ k",        "position": "phrase", "difficulty": 4, "type": "phrase"},
    {"word": "rainy morning", "phonemes": "r eɪ n i m ɔ r n ɪ ŋ",  "position": "phrase", "difficulty": 4, "type": "phrase"},
    {"word": "far from here", "phonemes": "f ɑ r f r ʌ m h ɪ r",   "position": "phrase", "difficulty": 4, "type": "phrase"},
    {"word": "wrong road",    "phonemes": "r ɔ ŋ r oʊ d",          "position": "phrase", "difficulty": 4, "type": "phrase"},
]

# Backward-compatible alias
WORDS = EXERCISES

_EXERCISES_BY_NAME = {e["word"]: e for e in EXERCISES}

# Levels as human-readable labels (for UI display)
LEVEL_LABELS = {
    0: "Level 1 · Syllables",
    1: "Level 2 · Starting Words",
    2: "Level 3 · Middle & End",
    3: "Level 4 · Blends & Vocalic",
    4: "Level 5 · Phrases",
}

LEVEL_DESCRIPTIONS = {
    0: "Build the /r/ sound in simple syllables",
    1: "Practice /r/ at the start of words",
    2: "Tackle /r/ in the middle and end of words",
    3: "Master /r/ blends and vowel combinations",
    4: "Carry your /r/ into real phrases",
}

XP_PER_CORRECT = 10
XP_FIRST_TRY_BONUS = 5      # extra XP if correct on the first attempt
ADVANCE_AFTER_CORRECT = 5   # consecutive correct to auto-advance level


def find_word(name: str | None) -> dict | None:
    if not name:
        return None
    return _EXERCISES_BY_NAME.get(name.strip().lower())


def get_next_word(
    history: list,
    suggestion: str = "",
    explicit_level: int | None = None,
) -> dict:
    """Pick the next exercise.

    Args:
        history:        Full session history list.
        suggestion:     "easier" | "harder" | "same" | "" — hint for difficulty.
        explicit_level: If provided, override the automatic progression and
                        pick from this curriculum level (0–4).
    """
    if explicit_level is not None:
        target_diff = max(0, min(4, explicit_level))
    else:
        n = len(history) if history else 0
        base_diff = min(3, 1 + n // 3)
        if suggestion == "easier":
            target_diff = max(0, base_diff - 1)
        elif suggestion == "harder":
            target_diff = min(4, base_diff + 1)
        else:
            target_diff = base_diff

    recent_words = {h.get("target_word") for h in (history[-5:] if history else [])}
    last_word = history[-1].get("target_word") if history else None

    pool = [e for e in EXERCISES if e["difficulty"] == target_diff and e["word"] not in recent_words]
    if not pool:
        pool = [e for e in EXERCISES if e["difficulty"] == target_diff and e["word"] != last_word]
    if not pool:
        pool = [e for e in EXERCISES if e["difficulty"] == target_diff]
    if not pool:
        pool = EXERCISES
    return random.choice(pool)
