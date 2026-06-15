"""Rhotic R Coach — /r/ practice app with guided curriculum progression.

Architecture overview
─────────────────────
• Five curriculum levels: Syllables → Starting Words → Middle/End →
  Blends/Vocalic → Phrases.
• XP is awarded for each correct attempt (10 pts + 5 bonus for first-try).
• After 5 consecutive correct answers at the current level the app offers
  an automatic level-up.
• "Hear it" button plays a clean TTS pronunciation of the target before
  the user attempts it — auditory model is a core SLP technique.
• Speech scoring runs locally; the coach can use Nemotron through the HF
  inference router when a Hugging Face token is configured.

User flow
─────────
The app boots into a HOME view: sign-in card (Hugging Face OAuth), a
resume card if the signed-in user has prior progress, and a five-button
level picker. Picking a level (or hitting Resume) switches into the
PRACTICE view, which is the original session loop. The Home button in
the practice view returns to the picker; progress is saved to a HF
Dataset on every attempt and every back-to-home transition.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import time

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Preload models at startup
# ---------------------------------------------------------------------------
print("[startup] Loading wav2vec2 phoneme model...")
import scoring
scoring._get_model()
print("[startup] wav2vec2 ready.")

print("[startup] Loading pocket-tts...")
import tts as tts_mod
tts_mod._load_engine()
tts_mod._load_state(tts_mod.DEFAULT_VOICE)
print("[startup] Warming TTS...")
for _ in tts_mod.speak_stream("Ready."):
    pass
print("[startup] TTS ready.")

import gradio as gr

import coach
from coach import coach_turn
from progress import load_progress, save_progress

# When COACH_BACKEND=local, pull the GGUF + warm the llama.cpp handle now
# instead of paying the cost on the first user turn. Matches wav2vec2 and
# pocket-tts above. Router backend is a no-op aside from a token sanity
# probe.
print(f"[startup] Preloading coach backend ({coach.BACKEND})...")
coach.preload()
print("[startup] coach backend ready.")
from scoring import score_pronunciation
from words import (
    get_next_word,
    LEVEL_LABELS, LEVEL_DESCRIPTIONS,
    LEVEL_UNLOCK_XP, level_unlocked,
    XP_PER_CORRECT, XP_FIRST_TRY_BONUS, ADVANCE_AFTER_CORRECT,
)
from twisters import todays_twister


# Short labels for the path nodes — drop the "Level N · " prefix from LEVEL_LABELS
LEVEL_SHORT_LABELS = {
    0: "Syllables",
    1: "Starting Words",
    2: "Middle &amp; End",
    3: "Blends &amp; Vocalic",
    4: "Phrases",
}


# Plain-text version of level labels, for TTS (drops the "·" character).
LEVEL_LABELS_TTS = {
    0: "Syllables",
    1: "Starting Words",
    2: "Middle and End",
    3: "Blends and Vocalic R",
    4: "Phrases",
}

HF_TOKEN_PRESENT = bool(os.environ.get("HF_TOKEN"))
DISCLAIMER = (
    "Rhotic R Coach is a practice tool, not a substitute for a licensed "
    "Speech-Language Pathologist."
)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(HERE, "assets")
gr.set_static_paths(paths=[ASSETS_DIR])


def _asset_url(filename: str) -> str:
    return f"/gradio_api/file={os.path.join(ASSETS_DIR, filename)}"


CELEBRATION_URLS = [
    _asset_url("jump.webm"),
    _asset_url("dab.webm"),
    _asset_url("hiphop.webm"),
    _asset_url("running_jump.webm"),
    _asset_url("box_jump.webm"),
]

with open(os.path.join(HERE, "theme.css")) as f:
    CUSTOM_CSS = f.read()
BLOB_IDLE_URL      = _asset_url("blob.svg")
BLOB_THINKING_URL  = _asset_url("blob-thinking.svg")
BLOB_CELEBRATE_URL = _asset_url("blob-celebrate.svg")

with open(os.path.join(HERE, "character.html")) as f:
    CHARACTER_HTML = (
        f.read()
        .replace("{{BLOB_URL}}", BLOB_IDLE_URL)
        .replace("{{BLOB_THINKING_URL}}", BLOB_THINKING_URL)
    )

# What counts as a "correct" attempt for XP, streaks, and level-up.
MIN_WORD_RECOGNIZABILITY = 0.40
ADVANCE_R_QUALITIES = {"correct", "approaching"}


def _is_attempt_correct(score: dict) -> bool:
    return (
        score.get("r_quality") in ADVANCE_R_QUALITIES
        and (score.get("phoneme_match") or 0.0) >= MIN_WORD_RECOGNIZABILITY
    )

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#FFFFFF",
    body_text_color="#0F2A4D",
    block_background_fill="#FFFFFF",
    block_border_color="#E2E8F0",
    block_border_width="0px",
    block_radius="20px",
    block_shadow="0 4px 24px rgba(15, 42, 77, 0.08)",
    button_primary_background_fill="#2563EB",
    button_primary_background_fill_hover="#1D4ED8",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#2563EB",
    button_large_radius="999px",
    button_small_radius="999px",
    input_background_fill="#F8FAFC",
    input_border_color="#E2E8F0",
    input_radius="14px",
)


# ---------------------------------------------------------------------------
# UI helper builders
# ---------------------------------------------------------------------------

def _word_html(word: str, ex_type: str = "word") -> str:
    type_badge = {
        "syllable": "<span class='rhotic-ex-badge syllable'>syllable drill</span>",
        "phrase":   "<span class='rhotic-ex-badge phrase'>phrase</span>",
        "word":     "<span class='rhotic-ex-badge word'>word</span>",
    }.get(ex_type, "")
    return f"<div class='rhotic-word-wrap'>{type_badge}<div class='rhotic-word'>{word}</div></div>"


def _ipa_html(phonemes: str) -> str:
    return f"<div class='rhotic-ipa'>/{phonemes}/</div>"


def _level_bar_html(level: int, xp: int, streak: int) -> str:
    label = LEVEL_LABELS.get(level, f"Level {level + 1}")
    desc  = LEVEL_DESCRIPTIONS.get(level, "")
    streak_badge = (
        f"<span class='rhotic-streak'>🔥 {streak} in a row</span>"
        if streak >= 2 else ""
    )

    at_max_level = level >= 4
    if at_max_level:
        progress_pct = 100
        progress_label = (
            f"🏆 Max level reached — keep practicing to extend your streak"
        )
    else:
        progress_pct = min(100, int(streak / ADVANCE_AFTER_CORRECT * 100))
        progress_label = f"{streak}/{ADVANCE_AFTER_CORRECT} correct to level up"

    return (
        f"<div class='rhotic-progress-bar'>"
        f"  <div class='rhotic-level-info'>"
        f"    <span class='rhotic-level-label'>{label}</span>"
        f"    <span class='rhotic-desc'>{desc}</span>"
        f"  </div>"
        f"  <div class='rhotic-xp-row'>"
        f"    <div class='rhotic-xp'>💫 {xp} XP</div>"
        f"    {streak_badge}"
        f"  </div>"
        f"  <div class='rhotic-progress-track'>"
        f"    <div class='rhotic-progress-fill' style='width:{progress_pct}%'></div>"
        f"  </div>"
        f"  <div class='rhotic-progress-label'>{progress_label}</div>"
        f"</div>"
    )


def _levelup_html(new_level: int) -> str:
    label = LEVEL_LABELS.get(new_level, f"Level {new_level + 1}")
    desc  = LEVEL_DESCRIPTIONS.get(new_level, "")
    return (
        f"<div class='rhotic-levelup'>"
        f"  <div class='rhotic-levelup-icon'>⬆️</div>"
        f"  <div class='rhotic-levelup-text'>Level up! Now: <strong>{label}</strong></div>"
        f"  <div class='rhotic-levelup-desc'>{desc}</div>"
        f"</div>"
    )


def _welcome_html(profile_username: str | None, profile_name: str | None) -> str:
    if profile_username:
        display = profile_name or profile_username
        return (
            f"<div class='rhotic-welcome-card'>"
            f"  <div class='rhotic-welcome-eyebrow'>Welcome back</div>"
            f"  <div class='rhotic-welcome-name'>{display}</div>"
            f"  <div class='rhotic-welcome-hint'>Your progress saves automatically.</div>"
            f"</div>"
        )
    return (
        f"<div class='rhotic-welcome-card'>"
        f"  <div class='rhotic-welcome-eyebrow'>Welcome</div>"
        f"  <div class='rhotic-welcome-name'>Practice the /r/ sound</div>"
        f"  <div class='rhotic-welcome-hint'>Sign in with Hugging Face to save progress across sessions.</div>"
        f"</div>"
    )


def _path_html(level: int, xp: int) -> str:
    """Render the winding 5-node level-picker path.

    `level` = the user's current/last-played level.
    `xp`    = accumulated XP across all sessions.

    Each node lands in one of four states:
      done    — n < level (user has moved past it)
      current — n == level
      open    — n > level and unlocked by XP
      locked  — XP < LEVEL_UNLOCK_XP[n]
    """
    # Zigzag offset in px applied via inline transform.
    OFFSETS = [-130, 0, 130, 0, -130]
    nodes = []
    for n in range(5):
        label    = LEVEL_SHORT_LABELS.get(n, f"Level {n + 1}")
        threshold = LEVEL_UNLOCK_XP.get(n, 0)
        unlocked = level_unlocked(n, xp)
        if not unlocked:
            state, icon, sub = "locked", "🔒", f"{threshold} XP to unlock"
        elif n == level:
            state, icon, sub = "current", "★", "You are here"
        elif n < level:
            state, icon, sub = "done", "✓", "Completed"
        else:
            state, icon, sub = "open", "●", "Practice"

        # We attach click behavior via a delegated listener (see demo.load js
        # below) instead of inline onclick — Gradio sanitizes inline handlers
        # off of dynamic gr.HTML updates, so onclick="..." silently disappears
        # the second time the path renders.
        nodes.append(
            f"<div class='rhotic-path-node {state}' "
            f"data-level='{n}' data-unlocked='{1 if unlocked else 0}' "
            f"style='transform: translateX({OFFSETS[n]}px)'>"
            f"  <div class='rhotic-node-circle'>{icon}</div>"
            f"  <div class='rhotic-node-label'>{label}</div>"
            f"  <div class='rhotic-node-sub'>{sub}</div>"
            f"</div>"
        )
    banner = (
        f"<div class='rhotic-path-banner'>"
        f"  <span class='chip'>💫 {xp} XP</span>"
        f"</div>"
    )
    return banner + f"<div class='rhotic-path'>{''.join(nodes)}</div>"


_SVG_OPEN  = ("<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' "
              "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>")
_SVG_CLOSE = "</svg>"
TAB_ICONS = {
    # Home / Learn
    "learn":   _SVG_OPEN + "<path d='M3 11 L12 3 L21 11'/>"
                            "<path d='M5 10v10h14V10'/>"
                            "<path d='M10 20v-6h4v6'/>" + _SVG_CLOSE,
    # Spiral (tongue twister)
    "twister": _SVG_OPEN + "<path d='M12 21a9 9 0 1 0-9-9'/>"
                            "<path d='M12 18a6 6 0 1 0-6-6'/>"
                            "<path d='M12 15a3 3 0 1 0-3-3'/>" + _SVG_CLOSE,
    # Person / Profile
    "profile": _SVG_OPEN + "<circle cx='12' cy='8' r='4'/>"
                            "<path d='M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6'/>" + _SVG_CLOSE,
}


def _sidebar_html(view: str) -> str:
    """Render the left sidebar with the active tab highlighted.

    `view` is the current view_state. When the user is in the practice
    flow we still mark "Learn" as active in the sidebar, since practice
    is reached from a Learn-tab path node.
    """
    active = "learn" if view in ("learn", "practice") else view
    tabs = [("learn", "Learn"), ("twister", "Twister"), ("profile", "Profile")]
    items = []
    for key, label in tabs:
        cls = "rhotic-tab active" if active == key else "rhotic-tab"
        icon = TAB_ICONS.get(key, "")
        items.append(
            f"<div class='{cls}' data-tab='{key}'>"
            f"  <span class='rhotic-tab-icon'>{icon}</span>"
            f"  <span class='rhotic-tab-label'>{label}</span>"
            f"</div>"
        )
    return (
        "<div class='rhotic-wordmark'>rhotic</div>"
        f"<nav>{''.join(items)}</nav>"
        "<div class='rhotic-side-footer'>"
        "  <button class='rhotic-about-link' "
        "          onclick=\"window.openRhoticAbout()\">About</button>"
        "</div>"
    )


def _twister_card_html(twister: dict) -> str:
    """Render the daily Twister hero card."""
    today_str = datetime.date.today().strftime("%A, %B %d")
    return (
        "<div class='rhotic-twister-card'>"
        "  <div class='rhotic-twister-eyebrow'>Today's Tongue Twister</div>"
        f" <div class='rhotic-twister-text'>{twister['text']}</div>"
        f" <div class='rhotic-twister-meta'>{today_str} · /r/ challenge</div>"
        "</div>"
    )


def _profile_html(profile, level: int, xp: int,
                  streak: int, best_streak: int, history: list) -> str:
    """Render the Profile tab.

    Signed out → sign-in CTA card. Signed in → name + stats + badges.
    The HF LoginButton itself lives below this HTML inside the Profile
    column (Gradio component), so this card just provides framing copy.
    """
    if not profile or not getattr(profile, "username", None):
        return (
            "<div class='rhotic-profile-card rhotic-profile-empty'>"
            "  <h2>Save your /r/ journey</h2>"
            "  <p>Sign in with Hugging Face to save XP, streaks, and the "
            "     levels you've completed across sessions.</p>"
            "</div>"
        )

    name    = getattr(profile, "name", None) or profile.username
    initial = (name or "?")[0].upper()
    attempts = len(history or [])
    badges = []
    for n in range(5):
        earned = level > n
        label = LEVEL_SHORT_LABELS.get(n, f"Lv {n+1}")
        icon  = "🏆" if earned else "🔒"
        cls   = "rhotic-badge earned" if earned else "rhotic-badge"
        badges.append(
            f"<div class='{cls}'>"
            f"  <span class='icon'>{icon}</span>{label}"
            f"</div>"
        )
    return (
        "<div class='rhotic-profile-card'>"
        "  <div class='rhotic-profile-header'>"
        f"   <div class='rhotic-profile-avatar'>{initial}</div>"
        "    <div>"
        f"     <div class='rhotic-profile-name'>{name}</div>"
        f"     <div class='rhotic-profile-handle'>@{profile.username}</div>"
        "    </div>"
        "  </div>"
        "  <div class='rhotic-stats-grid'>"
        f"   <div class='rhotic-stat'><div class='val'>{xp}</div>"
        "        <div class='lbl'>XP</div></div>"
        f"   <div class='rhotic-stat'><div class='val'>{best_streak}</div>"
        "        <div class='lbl'>Best streak</div></div>"
        f"   <div class='rhotic-stat'><div class='val'>{attempts}</div>"
        "        <div class='lbl'>Attempts</div></div>"
        "  </div>"
        "  <div class='rhotic-badges-title'>Levels achieved</div>"
        f"  <div class='rhotic-badges'>{''.join(badges)}</div>"
        "</div>"
    )


def _resume_card_html(progress: dict) -> str:
    if not progress.get("username") or progress.get("updated_at") is None:
        return ""
    level = int(progress.get("level", 0))
    xp    = int(progress.get("xp", 0))
    streak = int(progress.get("streak", 0))
    label = LEVEL_LABELS.get(level, f"Level {level + 1}")
    return (
        f"<div class='rhotic-resume-card'>"
        f"  <div class='rhotic-resume-label'>Continue your practice</div>"
        f"  <div class='rhotic-resume-stats'>"
        f"    <span class='rhotic-resume-level'>{label}</span>"
        f"    <span class='rhotic-resume-chip'>💫 {xp} XP</span>"
        f"    <span class='rhotic-resume-chip'>🔥 {streak} streak</span>"
        f"  </div>"
        f"</div>"
    )


def _wren_msg(text: str) -> dict:
    return {"role": "assistant", "content": f"🌀 {text}"}


def _user_msg(word: str) -> dict:
    return {"role": "user", "content": f"🎙️ (your attempt at \"{word}\")"}


# ---------------------------------------------------------------------------
# Coach + TTS helpers
# ---------------------------------------------------------------------------

_INTRO_TEMPLATES = {
    "syllable": "Let's start with '{target}'. Curl your tongue back as you say it.",
    "phrase":   "Try the phrase: '{target}'. Keep that R strong all the way through.",
    "word":     "Let's try '{target}'. Take your time with the R.",
}


def _intro_for_word(current_word: dict, history: list, messages: list):
    # The intro is shown the instant a level is picked, so it must be fast.
    # The LLM coach still runs on the feedback turn after the user records,
    # which is where it adds real value.
    exercise_type = current_word.get("type", "word")
    template = _INTRO_TEMPLATES.get(exercise_type, _INTRO_TEMPLATES["word"])
    reply = template.format(target=current_word["word"])
    sr, audio = tts_mod.synthesize_full(reply)
    new_messages = messages + [_wren_msg(reply)]
    return reply, (sr, audio), new_messages


def _tts_word(word: str) -> tuple[int, object]:
    """Synthesize just the target word for the 'Hear it' button."""
    return tts_mod.synthesize_full(word)


def _persist(username: str | None, level: int, xp: int, streak: int,
             best_streak: int, history: list) -> None:
    """Best-effort write to the HF Dataset. Never raises."""
    save_progress(username, {
        "level":       int(level),
        "xp":          int(xp),
        "streak":      int(streak),
        "best_streak": int(best_streak),
        "history":     history,
    })


# ---------------------------------------------------------------------------
# Home-view event handlers
# ---------------------------------------------------------------------------

def on_app_load(profile: gr.OAuthProfile | None):
    """Render the home view. Pull progress if the user is signed in."""
    username = profile.username if profile is not None else None
    name     = getattr(profile, "name", None) if profile is not None else None
    progress = load_progress(username) if username else {}
    welcome_html = _welcome_html(username, name)
    resume_html  = _resume_card_html(progress) if progress.get("username") else ""
    has_resume   = bool(resume_html)
    saved_level  = int(progress.get("level", 0))
    saved_xp     = int(progress.get("xp", 0))
    saved_streak = int(progress.get("streak", 0))
    saved_best   = int(progress.get("best_streak", saved_streak))
    saved_hist   = list(progress.get("history", []))
    return (
        # view-state machinery — boot into Learn (the path)
        "learn",                                  # view_state
        gr.update(visible=True),                  # learn_view (home_view)
        gr.update(visible=False),                 # twister_view
        gr.update(visible=False),                 # profile_view
        gr.update(visible=False),                 # practice_view
        _sidebar_html("learn"),                   # sidebar_display
        # learn-tab UI
        welcome_html,                             # welcome_display
        resume_html,                              # resume_display
        gr.update(visible=has_resume),            # resume_btn
        _path_html(saved_level, saved_xp),        # path_display
        # profile tab pre-rendered so it's ready when the user clicks
        _profile_html(profile, saved_level, saved_xp,
                      saved_streak, saved_best, saved_hist),  # profile_display
        # restored state for resume
        saved_level, saved_xp, saved_streak, saved_best, saved_hist,
        # username remembered through the session
        username or "",
    )


def _start_session(level: int, history: list, xp: int, streak: int, best_streak: int):
    """Shared bootstrap when entering the practice view at a given level."""
    first = get_next_word(history, "", explicit_level=level)
    reply, intro_audio, messages = _intro_for_word(first, history, [])
    return (
        # view-state
        "practice",
        gr.update(visible=False),              # learn_view
        gr.update(visible=False),              # twister_view
        gr.update(visible=False),              # profile_view
        gr.update(visible=True),               # practice_view
        _sidebar_html("practice"),             # sidebar (Learn stays highlighted)
        # practice state
        first,                                 # word_state
        history,                               # history_state
        messages,                              # transcript_state
        level, xp, streak, best_streak,        # level/xp/streak/best
        # rendered displays
        _word_html(first["word"], first.get("type", "word")),
        _ipa_html(first["phonemes"]),
        _level_bar_html(level, xp, streak),
        intro_audio,
        messages,
        gr.update(value=None),                 # clear mic
        reply,                                 # bubble bridge
        "false",                               # correct bridge
        "false",                               # ready bridge
        "",                                    # clear levelup notif
    )


def _load_session_state(profile: gr.OAuthProfile | None) -> dict:
    """Pull the user's saved progress from the HF Dataset at click time.

    This is the single source of truth for resume / level-pick — we always
    re-fetch instead of trusting whatever stale gr.State values are sitting
    in the browser. Without this the session resets every time the user
    bounces between Home and Practice.
    """
    username = profile.username if profile is not None else None
    if not username:
        return {"level": 0, "xp": 0, "streak": 0, "best_streak": 0, "history": []}
    progress = load_progress(username) or {}
    return {
        "level":       int(progress.get("level", 0)),
        "xp":          int(progress.get("xp", 0)),
        "streak":      int(progress.get("streak", 0)),
        "best_streak": int(progress.get("best_streak", 0)),
        "history":     list(progress.get("history", [])),
    }


def on_pick_level(level: int, current_xp: int,
                  profile: gr.OAuthProfile | None):
    """User tapped a path node. Keep accumulated XP, best_streak, and
    history — just jump to the chosen level.

    `current_xp` is the live xp_state value, which may have been bumped
    by the demo dev panel beyond what's persisted in the dataset. We
    take whichever is larger for the unlock check + the session bootstrap.
    """
    saved = _load_session_state(profile)
    effective_xp = max(int(current_xp or 0), int(saved["xp"]))
    if not level_unlocked(level, effective_xp):
        # Locked. No state change — stay on the current view. Returns one
        # gr.update() per output slot in start_session_outputs (23).
        return tuple(gr.update() for _ in range(23))
    return _start_session(
        level=level,
        history=saved["history"],
        xp=effective_xp,
        streak=saved["streak"],
        best_streak=saved["best_streak"],
    )


def on_resume(profile: gr.OAuthProfile | None):
    """User tapped 'Resume where you left off'. Replay everything from
    the dataset, including the level they were last on."""
    saved = _load_session_state(profile)
    return _start_session(
        level=saved["level"],
        history=saved["history"],
        xp=saved["xp"],
        streak=saved["streak"],
        best_streak=saved["best_streak"],
    )


def on_back_home(username: str, level: int, xp: int, streak: int,
                 best_streak: int, history: list,
                 profile: gr.OAuthProfile | None):
    """Practice → Home. Persist before switching."""
    effective_username = (profile.username if profile is not None else None) or username
    if effective_username:
        _persist(effective_username, level, xp, streak, best_streak, history)
    resume_html = _resume_card_html({
        "username":   effective_username,
        "updated_at": "now",
        "level":      level,
        "xp":         xp,
        "streak":     streak,
    }) if effective_username else ""
    return (
        "learn",
        gr.update(visible=True),               # learn_view
        gr.update(visible=False),              # twister_view
        gr.update(visible=False),              # profile_view
        gr.update(visible=False),              # practice_view
        _sidebar_html("learn"),                # sidebar
        resume_html,                           # resume_display
        gr.update(visible=bool(resume_html)),  # resume_btn
        _path_html(level, xp),                 # path_display
    )


# ---------------------------------------------------------------------------
# Practice-view event handlers
# ---------------------------------------------------------------------------

def on_hear_it(current_word):
    """Play a clean TTS pronunciation of the target word/phrase."""
    if not current_word:
        return gr.update()
    sr, audio = _tts_word(current_word["word"])
    return (sr, audio)


@coach.spaces.GPU(duration=120)
def on_submit(audio_path, current_word, history, messages,
              level, xp, streak, best_streak, username,
              profile: gr.OAuthProfile | None):
    if not current_word or not audio_path:
        return (current_word, history, messages, level, xp, streak, best_streak,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update())

    word = current_word["word"]
    target_phonemes = current_word["phonemes"]
    ex_type = current_word.get("type", "word")

    try:
        score = score_pronunciation(audio_path, word, target_phonemes)
    except Exception as e:
        print(f"[submit] scoring error: {e}")
        err = "Couldn't hear that clearly — try recording again."
        sr, audio = tts_mod.synthesize_full(err)
        new_msgs = messages + [_wren_msg(err)]
        return (current_word, history, new_msgs, level, xp, streak, best_streak,
                (sr, audio), new_msgs, err, "false", "false",
                _level_bar_html(level, xp, streak), gr.update(), gr.update())

    print(
        f"[score] word={word!r} type={ex_type} "
        f"r_quality={score['r_quality']} error={score['error_detail']} "
        f"f3={score['f3_hz']} r_seg={score['r_start_s']}-{score['r_end_s']}s "
        f"overall={score['overall_score']:.2f}"
    )

    state = {
        "current_target_word": word,
        "exercise_type": ex_type,
        "history": history,
    }
    coach_result = coach_turn(state, score, None)
    reply = coach_result["spoken_reply"]
    high_score = _is_attempt_correct(score)
    is_correct = high_score or bool(coach_result.get("is_correct"))

    # Natural "thinking" pause so the blob's thinking pose has time to
    # show before Wren speaks. The Gradio audio chain set the state to
    # "thinking" via _lock before this handler ran — sleeping here keeps
    # that state visible for the user.
    time.sleep(random.uniform(2.0, 4.0))

    is_first_try = not any(h.get("target_word") == word for h in history)
    new_xp = xp
    new_streak = streak
    if high_score:
        new_xp += XP_PER_CORRECT
        if is_first_try:
            new_xp += XP_FIRST_TRY_BONUS
        new_streak = streak + 1
    else:
        new_streak = 0

    new_best = max(int(best_streak or 0), new_streak)

    new_level = level
    levelup_notif = ""
    if new_streak >= ADVANCE_AFTER_CORRECT and level < 4:
        new_level = level + 1
        new_streak = 0
        levelup_notif = _levelup_html(new_level)
        lu_msg = (
            f"Level up — you've nailed this level. "
            f"Moving on to {LEVEL_LABELS_TTS[new_level]}."
        )
        reply = lu_msg
        sr, audio = tts_mod.synthesize_full(lu_msg)
    else:
        sr, audio = tts_mod.synthesize_full(reply)

    new_history = history + [{
        "target_word":      word,
        "exercise_type":    ex_type,
        "overall_score":    score["overall_score"],
        "r_quality":        score["r_quality"],
        "error_detail":     score["error_detail"],
        "detected_phonemes":score["detected_phonemes"],
        "f3_hz":            score["f3_hz"],
        "is_correct":       is_correct,
        "timestamp":        time.time(),
    }]

    new_messages = messages + [_user_msg(word), _wren_msg(reply)]

    effective_username = (profile.username if profile is not None else None) or username
    if effective_username:
        _persist(effective_username, new_level, new_xp, new_streak, new_best, new_history)

    # Each correct answer gets a unique celebrate-token so the JS watcher
    # sees a change every time (and re-fires the confetti).
    celebrate_value = f"go-{time.time()}" if high_score else ""

    return (
        current_word,
        new_history,
        new_messages,
        new_level,
        new_xp,
        new_streak,
        new_best,
        (sr, audio),                              # wren_audio
        new_messages,                             # transcript chatbot
        reply,                                    # bubble bridge
        "true" if high_score else "false",        # correct bridge
        "true" if high_score else "false",        # ready bridge
        _level_bar_html(new_level, new_xp, new_streak),
        levelup_notif,
        celebrate_value,                          # celebrate bridge
    )


def on_next(history, messages, level, xp, streak, best_streak):
    nxt = get_next_word(history, "", explicit_level=level)
    reply, intro_audio, new_messages = _intro_for_word(nxt, history, messages)
    return (
        nxt,
        history,
        new_messages,
        level, xp, streak, best_streak,
        _word_html(nxt["word"], nxt.get("type", "word")),
        _ipa_html(nxt["phonemes"]),
        _level_bar_html(level, xp, streak),
        intro_audio,
        new_messages,
        gr.update(value=None),                    # clear mic
        reply,                                    # bubble bridge
        "false",                                  # correct bridge
        "false",                                  # ready bridge
        "",                                       # clear level-up notif
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="Rhotic R Coach") as demo:

    # ---- Persistent state ----
    view_state           = gr.State("learn")
    word_state           = gr.State()
    twister_state        = gr.State()   # today's twister dict, populated lazily
    history_state        = gr.State([])
    transcript_state     = gr.State([])
    level_state          = gr.State(0)
    xp_state             = gr.State(0)
    streak_state         = gr.State(0)
    best_streak_state    = gr.State(0)
    remembered_user_state = gr.State("")

    # ---- Hidden JS bridges ----
    # Gradio 5 fully unmounts `visible=False` components — they vanish
    # from the DOM, so JS can't read their value. Instead we keep them
    # mounted with `visible=True` and hide them via CSS (off-screen).
    state_bridge     = gr.Textbox(value="idle",  elem_id="rhotic-state-bridge",
                                  show_label=False,
                                  elem_classes=["rhotic-bridge-hidden"])
    bubble_bridge    = gr.Textbox(value="",      elem_id="rhotic-bubble-bridge",
                                  show_label=False,
                                  elem_classes=["rhotic-bridge-hidden"])
    correct_bridge   = gr.Textbox(value="false", elem_id="rhotic-correct-bridge",
                                  show_label=False,
                                  elem_classes=["rhotic-bridge-hidden"])
    ready_bridge     = gr.Textbox(value="false", elem_id="rhotic-ready-bridge",
                                  show_label=False,
                                  elem_classes=["rhotic-bridge-hidden"])
    celebrate_bridge = gr.Textbox(value="",      elem_id="rhotic-celebrate-bridge",
                                  show_label=False,
                                  elem_classes=["rhotic-bridge-hidden"])

    # ---- Celebration modal (fixed-position overlay) ----
    # Triggered when the user gets an answer right or presses Shift+S.
    # Press Enter to dismiss. JS lives in the demo.load js hook.
    gr.HTML(
        f"<div id='rhotic-celebrate-modal' role='dialog' aria-modal='true'>"
        f"  <div class='rhotic-confetti'></div>"
        f"  <div class='rhotic-celebrate-text'>"
        f"    <div class='rhotic-celebrate-title'>Nice work!</div>"
        f"    <div class='rhotic-celebrate-hint'>Press Enter to continue</div>"
        f"  </div>"
        f"  <img class='rhotic-celebrate-blob' "
        f"       src='{BLOB_CELEBRATE_URL}' alt='Celebration'>"
        f"</div>"
    )

    # ---- About modal (fixed-position overlay, sits at root of the page) ----
    gr.HTML(
        "<div id='rhotic-about-modal' class='rhotic-modal-backdrop' role='dialog' aria-modal='true'>"
        "  <div class='rhotic-modal'>"
        "    <h2>About rhotic</h2>"
        f"   <p>{DISCLAIMER}</p>"
        "    <p>Phoneme scoring: wav2vec2 (facebook/wav2vec2-lv-60-espeak-cv-ft) + "
        "       Praat formant analysis with CTC-guided /r/ segment isolation.<br>"
        "       Coaching: SLP-informed feedback library matched to the live scoring signal.<br>"
        "       Voice: Kyutai pocket-tts.</p>"
        "    <p>Sign in with Hugging Face to save your progress across sessions.</p>"
        "    <p>Built for the Build Small Hackathon, June 2026.</p>"
        "    <button class='rhotic-modal-close' onclick=\"window.closeRhoticAbout()\">Close</button>"
        "  </div>"
        "</div>"
    )

    # gr.LoginButton requires either a real Space environment or a local
    # HF_TOKEN to bootstrap its mock OAuth handler. When running locally
    # without a token, we replace it with a static badge so the app still
    # boots — OAuth-gated features (progress sync) just stay session-only.
    _running_on_space = bool(os.environ.get("SPACE_ID"))
    _has_hf_token     = bool(os.environ.get("HF_TOKEN", "").strip())

    # ============================================================
    # APP SHELL — left sidebar + right main content area
    # ============================================================
    with gr.Row(elem_classes=["rhotic-shell"]):

        # ---- SIDEBAR ----
        with gr.Column(elem_classes=["rhotic-sidebar"]):
            sidebar_display = gr.HTML(elem_id="rhotic-sidebar-wrap")
            # Hidden tab-trigger buttons (clicks bridged from the
            # rendered .rhotic-tab divs via the demo.load JS hook).
            with gr.Row(elem_classes=["rhotic-hidden-row"]):
                tab_btn_learn   = gr.Button("learn",   elem_id="hidden-tab-learn",
                                             elem_classes=["rhotic-hidden-btn"])
                tab_btn_twister = gr.Button("twister", elem_id="hidden-tab-twister",
                                             elem_classes=["rhotic-hidden-btn"])
                tab_btn_profile = gr.Button("profile", elem_id="hidden-tab-profile",
                                             elem_classes=["rhotic-hidden-btn"])

        # ---- MAIN CONTENT COLUMN ----
        with gr.Column(elem_classes=["rhotic-main-col"]):

            # ===== LEARN view (path of levels) =====
            with gr.Column(visible=True, elem_classes=["rhotic-learn"]) as home_view:
                welcome_display = gr.HTML(elem_id="rhotic-welcome-wrap")
                resume_display  = gr.HTML(elem_id="rhotic-resume-wrap")
                resume_btn      = gr.Button(
                    "▶ Resume where you left off",
                    variant="primary",
                    visible=False,
                    elem_id="resume-btn",
                    elem_classes=["rhotic-resume-btn"],
                )
                gr.HTML("<div class='rhotic-pick-title'>Your /r/ practice path</div>")
                path_display = gr.HTML(elem_id="rhotic-path-wrap")
                # Hidden gr.Buttons act as click bridges for the path nodes.
                with gr.Row(elem_classes=["rhotic-hidden-row"]):
                    level_btn_0 = gr.Button("L0", elem_id="hidden-level-btn-0",
                                             elem_classes=["rhotic-hidden-btn"])
                    level_btn_1 = gr.Button("L1", elem_id="hidden-level-btn-1",
                                             elem_classes=["rhotic-hidden-btn"])
                    level_btn_2 = gr.Button("L2", elem_id="hidden-level-btn-2",
                                             elem_classes=["rhotic-hidden-btn"])
                    level_btn_3 = gr.Button("L3", elem_id="hidden-level-btn-3",
                                             elem_classes=["rhotic-hidden-btn"])
                    level_btn_4 = gr.Button("L4", elem_id="hidden-level-btn-4",
                                             elem_classes=["rhotic-hidden-btn"])

            # ===== TWISTER view (daily tongue twister) =====
            # Same surface as the Practice view: today's twister at top,
            # blob character + side bubble, big circular Record button at
            # the bottom (with a small Listen icon). The bubble + record
            # button + celebration are all driven by the SHARED bridges,
            # so the same JS works in either view — only one view is
            # mounted at a time so the shared element IDs don't collide.
            with gr.Column(visible=False, elem_classes=["rhotic-twister"]) as twister_view:
                # Headline card with today's twister text + meta
                twister_display = gr.HTML(elem_id="rhotic-twister-wrap")

                # Blob character + side speech bubble (driven by bubble_bridge)
                gr.HTML(CHARACTER_HTML)

                # Hidden audio players (autoplay; off-screen via CSS)
                twister_wren_audio = gr.Audio(
                    label="", show_label=False,
                    interactive=False, autoplay=True, type="numpy",
                    elem_id="twister-wren-audio",
                    elem_classes=["rhotic-audio-hidden"],
                )
                twister_hear_audio = gr.Audio(
                    label="", show_label=False,
                    interactive=False, autoplay=True, type="numpy",
                    elem_id="twister-hear-audio",
                    elem_classes=["rhotic-audio-hidden"],
                )
                # Off-screen mic widget — clicked by the visible Record btn.
                twister_mic = gr.Audio(
                    sources=["microphone"], type="filepath",
                    label="", show_label=False,
                    elem_id="twister-mic",
                    elem_classes=["rhotic-audio-hidden"],
                )
                # Hidden Listen bridge button (twister-scoped).
                with gr.Row(elem_classes=["rhotic-hidden-row"]):
                    twister_hear_btn = gr.Button(
                        "Listen",
                        elem_id="twister-hear-btn",
                        elem_classes=["rhotic-hidden-btn"],
                    )

                # Visible action bar — Listen + big Record. No Next (the
                # twister is the same all day; user just retries).
                gr.HTML("""
<div class="rhotic-actions-row" id="rhotic-twister-actions">
  <button type="button" class="rhotic-icon-btn rhotic-listen-circle"
          data-rho-action="twister-listen" aria-label="Listen to twister">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M6 8.5a6.5 6.5 0 0 1 13 0c0 4-3 4-3 8a3 3 0 0 1-6 0v-1"/>
      <path d="M15 8.5a2.5 2.5 0 0 0-5 0v1.5a1.5 1.5 0 1 1 0 3"/>
    </svg>
  </button>
  <button type="button" class="rhotic-record-btn" id="rhotic-record-btn"
          data-rho-action="record" aria-label="Record">
    <span class="rhotic-record-icon-wrap" id="rhotic-record-icon-wrap">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="2" width="6" height="13" rx="3"/>
        <path d="M19 10v1a7 7 0 0 1-14 0v-1"/>
        <line x1="12" y1="18" x2="12" y2="22"/>
        <line x1="8"  y1="22" x2="16" y2="22"/>
      </svg>
    </span>
  </button>
</div>
""")

            # ===== PROFILE view (sign-in OR stats) =====
            with gr.Column(visible=False, elem_classes=["rhotic-profile"]) as profile_view:
                profile_display = gr.HTML(elem_id="rhotic-profile-wrap")
                if _running_on_space or _has_hf_token:
                    login_btn = gr.LoginButton(
                        value="Sign in with Hugging Face",
                        elem_classes=["rhotic-login-btn"],
                    )
                else:
                    gr.HTML(
                        "<div class='rhotic-profile-card' style='text-align:center;'>"
                        "  <p style='color: var(--rhotic-text-mute);'>🏡 Local mode — "
                        "  set <code>HF_TOKEN</code> to enable Hugging Face sign-in.</p>"
                        "</div>"
                    )

            # ===== PRACTICE view (the per-word session loop) =====
            # Minimal Duolingo-style layout:
            #   ┌─────────────────────────────────────┐
            #   │ [←]  Level · XP · streak progress   │   top row
            #   ├─────────────────────────────────────┤
            #   │              [ word ]               │
            #   │           [ BLOB CHARACTER ]        │
            #   │       (speech bubble overlay)       │
            #   │                                     │
            #   │  [🔊 Listen]            [🎙 RECORD] │   action bar
            #   └─────────────────────────────────────┘
            # The Gradio mic widget is rendered off-screen — a styled
            # HTML record button bridges to it via JS, keeping the layout
            # clean. Wren's audio + the target audio also stay hidden.
            with gr.Column(visible=False, elem_classes=["rhotic-practice"]) as practice_view:

                with gr.Row(elem_classes=["rhotic-practice-top"]):
                    back_btn = gr.Button(
                        "←",
                        elem_id="back-btn",
                        elem_classes=["rhotic-back-icon-btn"],
                    )
                    progress_display = gr.HTML(elem_id="rhotic-progress-wrap")

                levelup_display  = gr.HTML(elem_id="rhotic-levelup-wrap")

                # Centered word + blob character
                word_display = gr.HTML(elem_id="rhotic-word-display")
                ipa_display  = gr.HTML(elem_id="rhotic-ipa-display")
                gr.HTML(CHARACTER_HTML)

                # Hidden audio players — autoplay but no visible UI.
                wren_audio = gr.Audio(
                    label="", show_label=False,
                    interactive=False, autoplay=True, type="numpy",
                    elem_id="wren-audio",
                    elem_classes=["rhotic-audio-hidden"],
                )
                hear_audio = gr.Audio(
                    label="", show_label=False,
                    interactive=False, autoplay=True, type="numpy",
                    elem_id="hear-audio",
                    elem_classes=["rhotic-audio-hidden"],
                )
                # Chatbot kept for state but visually hidden (transcript_state
                # still drives Wren's speech bubble via bubble_bridge).
                transcript = gr.Chatbot(
                    label="", show_label=False,
                    type="messages", height=1,
                    visible=False,
                )
                # Off-screen mic widget — clicked programmatically by the
                # custom Record button below.
                mic = gr.Audio(
                    sources=["microphone"], type="filepath",
                    label="", show_label=False,
                    elem_id="user-mic",
                    elem_classes=["rhotic-audio-hidden"],
                )

                # Hidden Gradio buttons — clicks bridged from the custom
                # HTML buttons below via window.rhoClickHidden().
                with gr.Row(elem_classes=["rhotic-hidden-row"]):
                    hear_btn = gr.Button(
                        "Listen",
                        elem_id="hear-btn",
                        elem_classes=["rhotic-hidden-btn"],
                    )
                    next_btn = gr.Button(
                        "Next",
                        elem_id="next-btn",
                        elem_classes=["rhotic-hidden-btn"],
                    )

                # Visible action bar: Listen (ear) · Record (mic) · Next (→).
                # Clicks are wired via data-rho-action + delegated listener
                # in the demo.load JS, since Gradio strips inline onclick
                # from dynamic gr.HTML updates.
                gr.HTML("""
<div class="rhotic-actions-row" id="rhotic-actions-row">
  <button type="button" class="rhotic-icon-btn rhotic-listen-circle"
          data-rho-action="listen" aria-label="Listen to target">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M6 8.5a6.5 6.5 0 0 1 13 0c0 4-3 4-3 8a3 3 0 0 1-6 0v-1"/>
      <path d="M15 8.5a2.5 2.5 0 0 0-5 0v1.5a1.5 1.5 0 1 1 0 3"/>
    </svg>
  </button>
  <button type="button" class="rhotic-record-btn" id="rhotic-record-btn"
          data-rho-action="record" aria-label="Record">
    <span class="rhotic-record-icon-wrap" id="rhotic-record-icon-wrap">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="2" width="6" height="13" rx="3"/>
        <path d="M19 10v1a7 7 0 0 1-14 0v-1"/>
        <line x1="12" y1="18" x2="12" y2="22"/>
        <line x1="8"  y1="22" x2="16" y2="22"/>
      </svg>
    </span>
  </button>
  <button type="button" class="rhotic-icon-btn rhotic-next-circle"
          data-rho-action="next" aria-label="Next word">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <line x1="5" y1="12" x2="19" y2="12"/>
      <polyline points="13 6 19 12 13 18"/>
    </svg>
  </button>
</div>
""")

    # ============================================================
    # DEV PANEL — hidden modal, toggled by Shift+D.
    # Lets the demo presenter edit XP / streak / level on the fly
    # so they can showcase unlocked levels without grinding XP.
    # ============================================================
    with gr.Column(elem_id="rhotic-dev-modal") as dev_modal:
        with gr.Column(elem_classes=["rhotic-dev-card"]):
            gr.HTML(
                "<h2>Demo controls</h2>"
                "<p>Edit XP, streak, or level on the fly, then Apply to "
                "re-render the path. Press <strong>Shift+D</strong> to close.</p>"
            )
            dev_xp     = gr.Number(label="XP",        value=0, precision=0, minimum=0)
            dev_streak = gr.Number(label="Streak",    value=0, precision=0, minimum=0)
            dev_level  = gr.Number(label="Level (0–4)", value=0, precision=0,
                                   minimum=0, maximum=4)
            with gr.Row():
                dev_apply  = gr.Button("Apply",      variant="primary")
                dev_unlock = gr.Button("Unlock all", variant="secondary")
            gr.HTML("<p class='rhotic-dev-hint'>Shift+D toggles this panel</p>")

    # ---------------------------------------------------------------------------
    # Output signatures
    # ---------------------------------------------------------------------------
    _common_state_outs = [
        word_state, history_state, transcript_state,
        level_state, xp_state, streak_state, best_streak_state,
    ]

    submit_outputs = _common_state_outs + [
        wren_audio, transcript,
        bubble_bridge, correct_bridge, ready_bridge,
        progress_display, levelup_display,
        celebrate_bridge,
    ]

    next_outputs = [
        word_state, history_state, transcript_state,
        level_state, xp_state, streak_state, best_streak_state,
        word_display, ipa_display, progress_display,
        wren_audio, transcript, mic,
        bubble_bridge, correct_bridge, ready_bridge,
        levelup_display,
    ]

    # All view-switch handlers share these five "shell" outputs in this exact
    # order: view_state, then the 4 view-column visibility updates, then the
    # sidebar HTML (so the active tab moves). Centralizing keeps the output
    # lists in sync as new views/tabs are added.
    _view_shell_outs = [
        view_state, home_view, twister_view, profile_view, practice_view,
        sidebar_display,
    ]

    start_session_outputs = _view_shell_outs + [
        word_state, history_state, transcript_state,
        level_state, xp_state, streak_state, best_streak_state,
        word_display, ipa_display, progress_display,
        wren_audio, transcript, mic,
        bubble_bridge, correct_bridge, ready_bridge,
        levelup_display,
    ]

    app_load_outputs = _view_shell_outs + [
        welcome_display, resume_display, resume_btn, path_display,
        profile_display,
        level_state, xp_state, streak_state, best_streak_state, history_state,
        remembered_user_state,
    ]

    back_home_outputs = _view_shell_outs + [
        resume_display, resume_btn, path_display,
    ]

    # Tab-switch handlers — all share the shell outputs; per-tab extras
    # (twister card, profile card) get added in the wiring section below.
    tab_learn_outputs    = _view_shell_outs
    tab_twister_outputs  = _view_shell_outs + [twister_state, twister_display]
    tab_profile_outputs  = _view_shell_outs + [profile_display]

    def on_tab_learn():
        return (
            "learn",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            _sidebar_html("learn"),
        )

    def on_tab_twister(level):
        twister = todays_twister(level or 0)
        return (
            "twister",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            _sidebar_html("twister"),
            twister,                            # twister_state
            _twister_card_html(twister),        # twister_display
        )

    def on_tab_profile(level, xp, streak, best_streak, history,
                       profile: gr.OAuthProfile | None):
        return (
            "profile",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            _sidebar_html("profile"),
            _profile_html(profile, level, xp, streak, best_streak, history),
        )

    # Twister submission — same scoring pipeline as Practice, no XP/streak.
    # Pushes to shared bubble_bridge (Wren's text) and celebrate_bridge
    # (confetti on correct) so the Practice JS works here unchanged.
    def on_twister_hear(twister):
        if not twister:
            return gr.update(), gr.update()
        sr, audio = _tts_word(twister["text"])
        # Mirror the twister text into the bubble while we play it.
        return (sr, audio), f"Listen: \"{twister['text']}\""

    def on_twister_submit(audio_path, twister):
        # 4 outputs: twister_wren_audio, bubble_bridge, celebrate_bridge, twister_mic
        if not twister or not audio_path:
            return gr.update(), gr.update(), gr.update(), gr.update()
        try:
            score = score_pronunciation(
                audio_path, twister["text"], twister["phonemes"],
            )
        except Exception as e:
            print(f"[twister] scoring error: {e}")
            err = "Couldn't hear that clearly — try recording again."
            sr, audio = tts_mod.synthesize_full(err)
            return (sr, audio), err, "", gr.update(value=None)

        # Natural thinking pause (matches Practice — the blob's thinking
        # pose has time to be visible thanks to the state_bridge flow).
        time.sleep(random.uniform(2.0, 4.0))

        quality = score.get("r_quality")
        if quality == "correct":
            reply = "Nice work — your /r/ came through clearly!"
        elif quality == "approaching":
            reply = "Getting there. Your /r/ is forming but not locked in yet."
        else:
            reply = "Let me hear that again. Focus on the /r/ — try slowing down."
        sr, audio = tts_mod.synthesize_full(reply)
        high_score = quality == "correct"
        celebrate = f"go-{time.time()}" if high_score else ""
        return (sr, audio), reply, celebrate, gr.update(value=None)

    # ---------------------------------------------------------------------------
    # Lock / unlock helpers
    # ---------------------------------------------------------------------------
    def _lock():
        return (gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False), "thinking")

    def _unlock():
        return (gr.update(interactive=True), gr.update(interactive=True),
                gr.update(interactive=True), "idle")

    lock_outputs   = [mic, hear_btn, next_btn, state_bridge]
    unlock_outputs = lock_outputs

    # ---------------------------------------------------------------------------
    # Wire events
    # ---------------------------------------------------------------------------

    # App boot — render home, populate welcome + resume from progress dataset.
    demo.load(on_app_load, inputs=None, outputs=app_load_outputs)

    # Sidebar tab clicks (bridged from .rhotic-tab DOM nodes via JS).
    tab_btn_learn.click(
        on_tab_learn, inputs=None, outputs=tab_learn_outputs,
    )
    tab_btn_twister.click(
        on_tab_twister, inputs=[level_state], outputs=tab_twister_outputs,
    )
    tab_btn_profile.click(
        on_tab_profile,
        inputs=[level_state, xp_state, streak_state,
                best_streak_state, history_state],
        outputs=tab_profile_outputs,
    )

    # Twister tab — Hear-it + mic recording → score → bubble + celebrate.
    # Uses shared bridges so the Practice JS handles bubble + thinking +
    # celebration without any twister-specific wiring on the front end.
    twister_hear_btn.click(
        on_twister_hear,
        inputs=[twister_state],
        outputs=[twister_hear_audio, bubble_bridge],
    )

    def _twister_lock():
        return (gr.update(interactive=False), gr.update(interactive=False),
                "thinking")

    def _twister_unlock():
        return (gr.update(interactive=True), gr.update(interactive=True),
                "idle")

    _twister_lock_outs = [twister_mic, twister_hear_btn, state_bridge]

    twister_mic.start_recording(
        lambda: "listening", inputs=None, outputs=[state_bridge],
    )
    twister_mic.stop_recording(
        _twister_lock, inputs=None, outputs=_twister_lock_outs,
    ).then(
        on_twister_submit,
        inputs=[twister_mic, twister_state],
        outputs=[twister_wren_audio, bubble_bridge, celebrate_bridge, twister_mic],
    ).then(
        _twister_unlock, inputs=None, outputs=_twister_lock_outs,
    )

    # JS bridge for path clicks + Shift+D dev panel toggle. Injected via the
    # demo.load `js` parameter (Gradio's intended way to run code on load).
    # Inline onclick="..." in gr.HTML gets stripped on dynamic updates, so we
    # use a single delegated click listener on document instead.
    demo.load(
        None, None, None,
        js=r"""
() => {
  if (window.__rhoticWired) return;
  window.__rhoticWired = true;

  // Helper: click a hidden Gradio button by elem_id (also exposed as
  // window.rhoClickHidden so inline onclick handlers can use it).
  function rhoClickHidden(id) {
    const wrap = document.getElementById(id);
    if (!wrap) return;
    const btn = wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button');
    if (btn) btn.click();
  }
  window.rhoClickHidden = rhoClickHidden;

  // Path-node click → trigger the matching hidden Gradio button.
  document.addEventListener('click', function(e) {
    const node = e.target.closest('.rhotic-path-node');
    if (!node) return;
    if (node.dataset.unlocked !== '1') return;
    rhoClickHidden('hidden-level-btn-' + node.dataset.level);
  });

  // Sidebar tab click → trigger the matching hidden tab button.
  document.addEventListener('click', function(e) {
    const tab = e.target.closest('.rhotic-tab[data-tab]');
    if (!tab) return;
    rhoClickHidden('hidden-tab-' + tab.dataset.tab);
  });

  // Practice / Twister action-row buttons (Listen / Record / Next).
  document.addEventListener('click', function(e) {
    const btn = e.target.closest('button[data-rho-action]');
    if (!btn) return;
    const action = btn.dataset.rhoAction;
    if (action === 'listen')         rhoClickHidden('hear-btn');
    else if (action === 'next')      rhoClickHidden('next-btn');
    else if (action === 'twister-listen') rhoClickHidden('twister-hear-btn');
    else if (action === 'record')    window.rhoticToggleMic();
  });

  // Big custom Record button → clicks Gradio audio's internal record
  // button so we can fully restyle the mic UI without forking Gradio.
  // We also flip the visual state immediately for instant feedback;
  // the state_bridge watcher will reconcile shortly after.
  window.rhoticToggleMic = function() {
    // Either the Practice mic ('user-mic') or the Twister mic is mounted
    // at any given time; one of them will be in the DOM.
    const wrap = document.getElementById('user-mic')
              || document.getElementById('twister-mic');
    if (!wrap) return;
    const inner =
      wrap.querySelector('button.record-button') ||
      wrap.querySelector('button[aria-label*="record" i]') ||
      wrap.querySelector('button[aria-label*="stop" i]') ||
      wrap.querySelector('button');
    if (!inner) return;
    const recBtn = document.getElementById('rhotic-record-btn');
    // Block taps while the model is scoring — Gradio audio is busy too.
    if (recBtn && recBtn.classList.contains('thinking')) return;
    inner.click();
    if (recBtn) {
      if (recBtn.classList.contains('recording')) {
        recBtn.classList.remove('recording');
        setRecordIcon(recBtn, 'idle');
      } else {
        recBtn.classList.add('recording');
        setRecordIcon(recBtn, 'recording');
      }
    }
  };
  function setRecordIcon(btn, state) {
    const iconWrap = btn.querySelector('.rhotic-record-icon-wrap');
    if (!iconWrap) return;
    if (state === 'recording')      iconWrap.innerHTML = STOP_SVG;
    else if (state === 'thinking')  iconWrap.innerHTML = HOURGLASS_SVG;
    else                            iconWrap.innerHTML = MIC_SVG;
  }

  // SVG icons used inside the big Record button — swapped in/out based
  // on state_bridge value so the user can see whether to start or stop.
  const MIC_SVG =
    "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' " +
    "stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>" +
    "<rect x='9' y='2' width='6' height='13' rx='3'/>" +
    "<path d='M19 10v1a7 7 0 0 1-14 0v-1'/>" +
    "<line x1='12' y1='18' x2='12' y2='22'/>" +
    "<line x1='8' y1='22' x2='16' y2='22'/>" +
    "</svg>";
  const STOP_SVG =
    "<svg viewBox='0 0 24 24' fill='currentColor'>" +
    "<rect x='6' y='6' width='12' height='12' rx='2'/>" +
    "</svg>";
  const HOURGLASS_SVG =
    "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' " +
    "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>" +
    "<path d='M6 2h12M6 22h12M6 2v4a6 6 0 0 0 12 0V2M6 22v-4a6 6 0 0 1 12 0v4'/>" +
    "</svg>";

  // ---------- CELEBRATION MODAL ----------
  // Triggered by server (celebrate_bridge) on a correct answer, or by
  // Shift+S in the practice view. Press Enter to dismiss.
  const CONFETTI_COLORS = ['#3B82F6', '#22C55E', '#F59E0B', '#A78BFA', '#EC4899', '#F472B6'];
  function showCelebration() {
    const m = document.getElementById('rhotic-celebrate-modal');
    if (!m) return;
    const confetti = m.querySelector('.rhotic-confetti');
    if (confetti) {
      confetti.innerHTML = '';
      for (let i = 0; i < 60; i++) {
        const p = document.createElement('div');
        p.className = 'rhotic-confetti-piece';
        p.style.left = (Math.random() * 100) + 'vw';
        p.style.background = CONFETTI_COLORS[Math.floor(Math.random() * CONFETTI_COLORS.length)];
        p.style.animationDuration = (2.2 + Math.random() * 2.2) + 's';
        p.style.animationDelay    = (Math.random() * 0.6) + 's';
        confetti.appendChild(p);
      }
    }
    m.classList.add('open');
  }
  function hideCelebration() {
    const m = document.getElementById('rhotic-celebrate-modal');
    if (m) m.classList.remove('open');
  }
  document.addEventListener('keydown', function(e) {
    // Enter closes celebration if open
    if (e.key === 'Enter') {
      const m = document.getElementById('rhotic-celebrate-modal');
      if (m && m.classList.contains('open')) {
        e.preventDefault();
        hideCelebration();
        return;
      }
    }
    // Shift+S celebration shortcut intentionally disabled in the main build.
  });
  // Watch celebrate_bridge for server-triggered celebrations.
  function watchCelebrate() {
    const box = document.getElementById('rhotic-celebrate-bridge');
    if (!box) { setTimeout(watchCelebrate, 200); return; }
    const input = box.querySelector('textarea, input');
    if (!input) { setTimeout(watchCelebrate, 200); return; }
    let last = input.value || '';
    setInterval(() => {
      const v = input.value || '';
      if (v && v !== last) { last = v; showCelebration(); }
      else if (!v) { last = ''; }
    }, 100);
  }
  watchCelebrate();

  // ---------- BUBBLE + BLOB POSE + RECORD BUTTON STATE ----------
  // One unified poller drives everything that mirrors a hidden bridge.
  // Inline <script> in gr.HTML gets stripped by Gradio 5's sanitizer, so
  // we can't rely on character.html for any JS — it all has to live here.
  function getInput(id) {
    const box = document.getElementById(id);
    if (!box) return null;
    return box.querySelector('textarea, input');
  }
  function setBubble(text) {
    const b = document.getElementById('rhotic-bubble');
    if (!b) return;
    if (text && text.trim()) {
      b.textContent = text;
      b.classList.remove('hidden');
    } else {
      b.classList.add('hidden');
    }
  }
  function setBlobPose(pose) {
    const img = document.getElementById('rhotic-blob');
    if (!img) return;
    const want = pose === 'thinking'
      ? img.dataset.rhoticThinking
      : img.dataset.rhoticIdle;
    if (want && img.getAttribute('src') !== want) {
      img.setAttribute('src', want);
    }
  }
  function setReadyBadge(show) {
    const b = document.getElementById('rhotic-ready-badge');
    if (!b) return;
    if (show) b.classList.remove('hidden');
    else b.classList.add('hidden');
  }
  function setRecordButton(state) {
    const btn = document.getElementById('rhotic-record-btn');
    const iconWrap = document.getElementById('rhotic-record-icon-wrap');
    if (!btn || !iconWrap) return;
    btn.classList.remove('recording', 'thinking');
    if (state === 'listening') {
      btn.classList.add('recording');
      iconWrap.innerHTML = STOP_SVG;
      btn.setAttribute('aria-label', 'Stop recording');
    } else if (state === 'thinking') {
      btn.classList.add('thinking');
      iconWrap.innerHTML = HOURGLASS_SVG;
      btn.setAttribute('aria-label', 'Scoring');
    } else {
      iconWrap.innerHTML = MIC_SVG;
      btn.setAttribute('aria-label', 'Record');
    }
  }

  let lastBubble = null, lastState = null, lastReady = null;
  setInterval(() => {
    const bubbleInput = getInput('rhotic-bubble-bridge');
    if (bubbleInput && bubbleInput.value !== lastBubble) {
      lastBubble = bubbleInput.value;
      setBubble(lastBubble);
    }
    const stateInput = getInput('rhotic-state-bridge');
    if (stateInput && stateInput.value !== lastState) {
      lastState = stateInput.value;
      setBlobPose(lastState === 'thinking' ? 'thinking' : 'idle');
      setRecordButton(lastState);
    }
    const readyInput = getInput('rhotic-ready-bridge');
    if (readyInput && readyInput.value !== lastReady) {
      lastReady = readyInput.value;
      setReadyBadge(lastReady === 'true');
    }
  }, 100);

  // Shift+D dev-panel toggle intentionally disabled in the main build.
}
""",
    )

    # Level pickers — re-load progress from dataset at click time, then
    # jump to the chosen level. OAuthProfile is auto-injected by Gradio.
    def _wire_level_btn(btn, level):
        btn.click(
            on_pick_level,
            inputs=[gr.State(level), xp_state],
            outputs=start_session_outputs,
        )
    _wire_level_btn(level_btn_0, 0)
    _wire_level_btn(level_btn_1, 1)
    _wire_level_btn(level_btn_2, 2)
    _wire_level_btn(level_btn_3, 3)
    _wire_level_btn(level_btn_4, 4)

    # Resume button — re-load from dataset, restore saved level/XP/streak.
    resume_btn.click(
        on_resume,
        inputs=None,
        outputs=start_session_outputs,
    )

    # Back to home
    back_btn.click(
        on_back_home,
        inputs=[
            remembered_user_state,
            level_state, xp_state, streak_state, best_streak_state, history_state,
        ],
        outputs=back_home_outputs,
    )

    # Mic state → listening
    mic.start_recording(lambda: "listening", inputs=None, outputs=[state_bridge])

    # Record → score → unlock
    mic.stop_recording(_lock, inputs=None, outputs=lock_outputs).then(
        on_submit,
        inputs=[mic, word_state, history_state, transcript_state,
                level_state, xp_state, streak_state, best_streak_state,
                remembered_user_state],
        outputs=submit_outputs,
    ).then(
        _unlock, inputs=None, outputs=unlock_outputs,
    )

    # "Hear it" button — lock, synthesize + show player, unlock
    hear_btn.click(
        _lock, inputs=None, outputs=lock_outputs,
    ).then(
        on_hear_it,
        inputs=[word_state],
        outputs=[hear_audio],
    ).then(
        _unlock, inputs=None, outputs=unlock_outputs,
    )

    # Next button
    next_btn.click(_lock, inputs=None, outputs=lock_outputs).then(
        on_next,
        inputs=[history_state, transcript_state,
                level_state, xp_state, streak_state, best_streak_state],
        outputs=next_outputs,
    ).then(
        _unlock, inputs=None, outputs=unlock_outputs,
    )

    # ---------------------------------------------------------------------------
    # Dev panel wiring — Apply / Unlock-all both push xp/streak/level into the
    # session state and re-render the path (and the practice progress bar, in
    # case the panel is opened mid-session).
    # ---------------------------------------------------------------------------
    def on_dev_apply(xp, streak, level, best_streak, history,
                     profile: gr.OAuthProfile | None):
        xp     = int(max(0, xp or 0))
        streak = int(max(0, streak or 0))
        level  = int(max(0, min(4, level or 0)))
        best   = max(int(best_streak or 0), streak)
        return (
            xp, streak, level, best,
            _path_html(level, xp),
            _level_bar_html(level, xp, streak),
            _profile_html(profile, level, xp, streak, best, history),
        )

    def on_dev_unlock_all(streak, level, best_streak, history,
                          profile: gr.OAuthProfile | None):
        xp = max(LEVEL_UNLOCK_XP.values()) + 100
        streak = int(max(0, streak or 0))
        level  = int(max(0, min(4, level or 0)))
        best   = max(int(best_streak or 0), streak)
        return (
            xp, streak, level, best,
            _path_html(level, xp),
            _level_bar_html(level, xp, streak),
            _profile_html(profile, level, xp, streak, best, history),
        )

    _dev_outputs = [xp_state, streak_state, level_state, best_streak_state,
                    path_display, progress_display, profile_display]
    dev_apply.click(
        on_dev_apply,
        inputs=[dev_xp, dev_streak, dev_level, best_streak_state, history_state],
        outputs=_dev_outputs,
    )
    dev_unlock.click(
        on_dev_unlock_all,
        inputs=[dev_streak, dev_level, best_streak_state, history_state],
        outputs=_dev_outputs,
    )

    # About-modal open/close. Path clicks + Shift+D are wired via the
    # demo.load(js=...) hook above instead — that fires reliably whereas
    # script tags inside gr.HTML are sometimes skipped on dynamic updates.
    gr.HTML("""
    <script>
    (function initRhoticModal() {
      const modal = document.getElementById('rhotic-about-modal');
      if (!modal) { setTimeout(initRhoticModal, 200); return; }
      window.openRhoticAbout  = () => modal.classList.add('open');
      window.closeRhoticAbout = () => modal.classList.remove('open');
      modal.addEventListener('click', e => {
        if (e.target === modal) window.closeRhoticAbout();
      });
    })();
    </script>
    """)


if __name__ == "__main__":
    demo.launch()
