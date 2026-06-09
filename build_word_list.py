import os
import sys

from scoring import score_pronunciation

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(HERE, "recordings")
WORDS_FILE = os.path.join(HERE, "words.py")

# Candidate /r/ words organized by position of /r/ in the word and difficulty
# (1 = easiest, 3 = hardest). The clinical progression for rhotacism therapy
# generally goes: initial /r/ in stressed syllables -> medial -> final/vocalic
# -> consonant blends. Within each bucket, longer/multisyllabic words are
# harder.
CANDIDATES = [
    # Initial /r/
    {"word": "red", "position": "initial", "difficulty": 1},
    {"word": "run", "position": "initial", "difficulty": 1},
    {"word": "rain", "position": "initial", "difficulty": 1},
    {"word": "rope", "position": "initial", "difficulty": 1},
    {"word": "rock", "position": "initial", "difficulty": 1},
    {"word": "rabbit", "position": "initial", "difficulty": 2},
    {"word": "river", "position": "initial", "difficulty": 2},
    {"word": "radio", "position": "initial", "difficulty": 2},
    {"word": "rocket", "position": "initial", "difficulty": 2},
    # Medial /r/
    {"word": "very", "position": "medial", "difficulty": 2},
    {"word": "carrot", "position": "medial", "difficulty": 2},
    {"word": "story", "position": "medial", "difficulty": 2},
    {"word": "orange", "position": "medial", "difficulty": 3},
    {"word": "America", "position": "medial", "difficulty": 3},
    {"word": "tomorrow", "position": "medial", "difficulty": 3},
    # Final / vocalic /r/
    {"word": "car", "position": "final", "difficulty": 2},
    {"word": "four", "position": "final", "difficulty": 2},
    {"word": "door", "position": "final", "difficulty": 2},
    {"word": "star", "position": "final", "difficulty": 2},
    {"word": "teacher", "position": "final", "difficulty": 3},
    {"word": "doctor", "position": "final", "difficulty": 3},
    # Consonant blends with /r/
    {"word": "tree", "position": "blend", "difficulty": 3},
    {"word": "dress", "position": "blend", "difficulty": 3},
    {"word": "green", "position": "blend", "difficulty": 3},
    {"word": "bread", "position": "blend", "difficulty": 3},
    {"word": "crab", "position": "blend", "difficulty": 3},
    {"word": "frog", "position": "blend", "difficulty": 3},
    {"word": "brown", "position": "blend", "difficulty": 3},
    {"word": "grass", "position": "blend", "difficulty": 3},
    {"word": "truck", "position": "blend", "difficulty": 3},
]


WORDS_FILE_TEMPLATE = '''import random

WORDS = [
{words_block}]


def get_next_word(history: list, suggestion: str) -> dict:
    """Pick an appropriate next word given recent attempt history.

    suggestion in {{"same", "easier", "harder"}}. Falls back to "same difficulty"
    when given anything else or when history is empty.
    """
    by_word = {{w["word"]: w for w in WORDS}}

    if suggestion == "same" and history:
        last_word = history[-1].get("target_word")
        if last_word in by_word:
            return by_word[last_word]

    recent = history[-5:] if history else []
    recent_diffs = [
        by_word[h["target_word"]]["difficulty"]
        for h in recent
        if h.get("target_word") in by_word
    ]
    avg_diff = sum(recent_diffs) / len(recent_diffs) if recent_diffs else 1.0

    if suggestion == "easier":
        target_diff = max(1, int(round(avg_diff)) - 1)
    elif suggestion == "harder":
        target_diff = min(3, int(round(avg_diff)) + 1)
    else:
        target_diff = max(1, min(3, int(round(avg_diff))))

    recent_words = {{h.get("target_word") for h in recent}}
    pool = [w for w in WORDS if w["difficulty"] == target_diff and w["word"] not in recent_words]
    if not pool:
        pool = [w for w in WORDS if w["difficulty"] == target_diff]
    if not pool:
        pool = WORDS
    return random.choice(pool)
'''


def _format_words_block(approved: list) -> str:
    lines = []
    for w in approved:
        lines.append(
            "    {{\"word\": {word!r}, \"phonemes\": {phonemes!r}, "
            "\"position\": {position!r}, \"difficulty\": {difficulty}}},".format(**w)
        )
    return "\n".join(lines) + "\n"


def _write_words_py(approved: list) -> None:
    block = _format_words_block(approved)
    content = WORDS_FILE_TEMPLATE.format(words_block=block)
    with open(WORDS_FILE, "w") as f:
        f.write(content)


def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        resp = input(f"{msg}{suffix}: ").strip()
    except EOFError:
        return default
    return resp or default


def main() -> None:
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    print(f"Recordings directory: {RECORDINGS_DIR}")
    print(f"Place a .wav of each target word at recordings/<word>.wav before")
    print(f"or during this run. You can skip words you don't want to record.\n")

    approved = []
    for entry in CANDIDATES:
        word = entry["word"]
        audio_path = os.path.join(RECORDINGS_DIR, f"{word}.wav")

        print(f"\n--- {word} (position={entry['position']}, difficulty={entry['difficulty']}) ---")

        if not os.path.isfile(audio_path):
            action = _prompt(
                f"No recording at recordings/{word}.wav. (w)ait/(s)kip/(q)uit",
                default="s",
            ).lower()
            if action == "q":
                print("Quitting setup.")
                break
            if action == "w":
                input(f"Drop a recording at recordings/{word}.wav, then press Enter... ")
                if not os.path.isfile(audio_path):
                    print("  Still missing. Skipping.")
                    continue
            else:
                print("  Skipping.")
                continue

        try:
            result = score_pronunciation(audio_path, word, "")
        except Exception as e:
            print(f"  ERROR running model on {audio_path}: {e}")
            continue

        detected = result["detected_phonemes"]
        print(f"  Detected phonemes: {detected!r}")

        action = _prompt("(a)ccept/(e)dit/(s)kip", default="a").lower()
        if action == "s":
            print("  Skipping.")
            continue
        if action == "e":
            edited = _prompt("  Enter corrected phonemes (space-separated IPA)", default=detected)
            if edited:
                detected = edited

        approved.append(
            {
                "word": word,
                "phonemes": detected,
                "position": entry["position"],
                "difficulty": entry["difficulty"],
            }
        )
        print(f"  Added: {approved[-1]}")

    if not approved:
        print("\nNo approved words; not writing words.py.")
        return

    _write_words_py(approved)
    print(f"\nWrote {len(approved)} words to {WORDS_FILE}")


if __name__ == "__main__":
    main()
