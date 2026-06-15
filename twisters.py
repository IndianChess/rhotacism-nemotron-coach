"""Curated /r/ tongue twisters for the daily Twister tab.

Each entry has a `text` (what the user sees + records) and `phonemes` (the
target sequence consumed by `scoring.score_pronunciation`). Phonemes use the
same "r" convention as words.py, scoring normalizes ɹ → r.

Twisters are bucketed into easy/medium/hard. `todays_twister(level)` picks
deterministically from the bucket matching the user's curriculum level, so a
beginner gets a short twister and a level-5 user gets a brutal one. The pick
rotates daily within each bucket.
"""

from __future__ import annotations

import datetime
import hashlib


TWISTERS_BY_DIFFICULTY = {
    "easy": [
        {
            "text": "Red lorry, yellow lorry",
            "phonemes": "r ɛ d l ɔ r i j ɛ l oʊ l ɔ r i",
        },
        {
            "text": "Three free throws",
            "phonemes": "θ r iː f r iː θ r oʊ z",
        },
    ],
    "medium": [
        {
            "text": "The red wren ran round the rough river",
            "phonemes": "ð ə r ɛ d r ɛ n r æ n r aʊ n d ð ə r ʌ f r ɪ v ə r",
        },
        {
            "text": "Robert rapidly rolled the rolling rocks",
            "phonemes": "r ɑ b ə r t r æ p ɪ d l i r oʊ l d ð ə r oʊ l ɪ ŋ r ɑ k s",
        },
    ],
    "hard": [
        {
            "text": "Round the rugged rocks the ragged rascal ran",
            "phonemes": "r aʊ n d ð ə r ʌ g ə d r ɑ k s ð ə r æ g ə d r æ s k ə l r æ n",
        },
        {
            "text": "Rural ruler ruled rural areas",
            "phonemes": "r ʊ r ə l r u l ə r r u l d r ʊ r ə l ɛ r i ə z",
        },
        {
            "text": "Roberta ran rings around the Roman ruins",
            "phonemes": "r oʊ b ɜ r t ə r æ n r ɪ ŋ z ə r aʊ n d ð ə r oʊ m ə n r u ɪ n z",
        },
        {
            "text": "Around the rugged rocks the ragged rabbit ran",
            "phonemes": "ə r aʊ n d ð ə r ʌ g ə d r ɑ k s ð ə r æ g ə d r æ b ɪ t r æ n",
        },
    ],
}

# Flat list kept for any caller that still imports it (e.g. tests).
TWISTERS = (
    TWISTERS_BY_DIFFICULTY["easy"]
    + TWISTERS_BY_DIFFICULTY["medium"]
    + TWISTERS_BY_DIFFICULTY["hard"]
)


def _bucket_for_level(level: int) -> str:
    if level <= 1:
        return "easy"
    if level <= 3:
        return "medium"
    return "hard"


def todays_twister(level: int = 0, today: datetime.date | None = None) -> dict:
    """Pick today's twister deterministically by date and curriculum level.

    Same date + level → same twister. Difficulty buckets:
      levels 0-1 → easy, 2-3 → medium, 4+ → hard.
    """
    today = today or datetime.date.today()
    bucket = _bucket_for_level(level)
    pool = TWISTERS_BY_DIFFICULTY[bucket]
    digest = hashlib.md5(f"{today.isoformat()}|{bucket}".encode()).hexdigest()
    idx = int(digest, 16) % len(pool)
    return pool[idx]
