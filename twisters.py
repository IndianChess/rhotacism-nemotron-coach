"""Curated /r/ tongue twisters for the daily Twister tab.

Each entry has a `text` (what the user sees + records) and `phonemes` (the
target sequence consumed by `scoring.score_pronunciation`). Phonemes use the
same "r" convention as words.py — scoring normalizes ɹ → r.

`todays_twister()` is deterministic on the current calendar date, so every
user sees the same twister on a given day and it rotates at midnight local.
"""

from __future__ import annotations

import datetime
import hashlib


TWISTERS = [
    {
        "text": "Red lorry, yellow lorry",
        "phonemes": "r ɛ d l ɔ r i j ɛ l oʊ l ɔ r i",
    },
    {
        "text": "Round the rugged rocks the ragged rascal ran",
        "phonemes": "r aʊ n d ð ə r ʌ g ə d r ɑ k s ð ə r æ g ə d r æ s k ə l r æ n",
    },
    {
        "text": "Three free throws",
        "phonemes": "θ r iː f r iː θ r oʊ z",
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
        "text": "Robert rapidly rolled the rolling rocks",
        "phonemes": "r ɑ b ə r t r æ p ɪ d l i r oʊ l d ð ə r oʊ l ɪ ŋ r ɑ k s",
    },
    {
        "text": "The red wren ran round the rough river",
        "phonemes": "ð ə r ɛ d r ɛ n r æ n r aʊ n d ð ə r ʌ f r ɪ v ə r",
    },
    {
        "text": "Around the rugged rocks the ragged rabbit ran",
        "phonemes": "ə r aʊ n d ð ə r ʌ g ə d r ɑ k s ð ə r æ g ə d r æ b ɪ t r æ n",
    },
]


def todays_twister(today: datetime.date | None = None) -> dict:
    """Pick today's twister deterministically by date.

    Same date → same twister for every user on the planet.
    """
    today = today or datetime.date.today()
    digest = hashlib.md5(today.isoformat().encode()).hexdigest()
    idx = int(digest, 16) % len(TWISTERS)
    return TWISTERS[idx]
