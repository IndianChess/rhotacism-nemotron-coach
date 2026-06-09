"""Rivet R Coach — /r/ practice app with guided curriculum progression.

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

import json
import os
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

from coach import coach_turn
from progress import load_progress, save_progress
from scoring import score_pronunciation
from words import (
    get_next_word,
    LEVEL_LABELS, LEVEL_DESCRIPTIONS,
    XP_PER_CORRECT, XP_FIRST_TRY_BONUS, ADVANCE_AFTER_CORRECT,
)


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
    "Rivet R Coach is a practice tool, not a substitute for a licensed "
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
with open(os.path.join(HERE, "character.html")) as f:
    CHARACTER_HTML = (
        f.read()
        .replace("{{IDLE_URL}}", _asset_url("idle.webm"))
        .replace("{{CELEBRATIONS_JSON}}", json.dumps(CELEBRATION_URLS))
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
        "syllable": "<span class='rivet-ex-badge syllable'>syllable drill</span>",
        "phrase":   "<span class='rivet-ex-badge phrase'>phrase</span>",
        "word":     "<span class='rivet-ex-badge word'>word</span>",
    }.get(ex_type, "")
    return f"<div class='rivet-word-wrap'>{type_badge}<div class='rivet-word'>{word}</div></div>"


def _ipa_html(phonemes: str) -> str:
    return f"<div class='rivet-ipa'>/{phonemes}/</div>"


def _level_bar_html(level: int, xp: int, streak: int) -> str:
    label = LEVEL_LABELS.get(level, f"Level {level + 1}")
    desc  = LEVEL_DESCRIPTIONS.get(level, "")
    streak_badge = (
        f"<span class='rivet-streak'>🔥 {streak} in a row</span>"
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
        f"<div class='rivet-progress-bar'>"
        f"  <div class='rivet-level-info'>"
        f"    <span class='rivet-level-label'>{label}</span>"
        f"    <span class='rivet-desc'>{desc}</span>"
        f"  </div>"
        f"  <div class='rivet-xp-row'>"
        f"    <div class='rivet-xp'>💫 {xp} XP</div>"
        f"    {streak_badge}"
        f"  </div>"
        f"  <div class='rivet-progress-track'>"
        f"    <div class='rivet-progress-fill' style='width:{progress_pct}%'></div>"
        f"  </div>"
        f"  <div class='rivet-progress-label'>{progress_label}</div>"
        f"</div>"
    )


def _levelup_html(new_level: int) -> str:
    label = LEVEL_LABELS.get(new_level, f"Level {new_level + 1}")
    desc  = LEVEL_DESCRIPTIONS.get(new_level, "")
    return (
        f"<div class='rivet-levelup'>"
        f"  <div class='rivet-levelup-icon'>⬆️</div>"
        f"  <div class='rivet-levelup-text'>Level up! Now: <strong>{label}</strong></div>"
        f"  <div class='rivet-levelup-desc'>{desc}</div>"
        f"</div>"
    )


def _welcome_html(profile_username: str | None, profile_name: str | None) -> str:
    if profile_username:
        display = profile_name or profile_username
        return (
            f"<div class='rivet-welcome-card'>"
            f"  <div class='rivet-welcome-eyebrow'>Welcome back</div>"
            f"  <div class='rivet-welcome-name'>{display}</div>"
            f"  <div class='rivet-welcome-hint'>Your progress saves automatically.</div>"
            f"</div>"
        )
    return (
        f"<div class='rivet-welcome-card'>"
        f"  <div class='rivet-welcome-eyebrow'>Welcome</div>"
        f"  <div class='rivet-welcome-name'>Practice the /r/ sound</div>"
        f"  <div class='rivet-welcome-hint'>Sign in with Hugging Face to save progress across sessions.</div>"
        f"</div>"
    )


def _resume_card_html(progress: dict) -> str:
    if not progress.get("username") or progress.get("updated_at") is None:
        return ""
    level = int(progress.get("level", 0))
    xp    = int(progress.get("xp", 0))
    streak = int(progress.get("streak", 0))
    label = LEVEL_LABELS.get(level, f"Level {level + 1}")
    return (
        f"<div class='rivet-resume-card'>"
        f"  <div class='rivet-resume-label'>Continue your practice</div>"
        f"  <div class='rivet-resume-stats'>"
        f"    <span class='rivet-resume-level'>{label}</span>"
        f"    <span class='rivet-resume-chip'>💫 {xp} XP</span>"
        f"    <span class='rivet-resume-chip'>🔥 {streak} streak</span>"
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

def _intro_for_word(current_word: dict, history: list, messages: list):
    state = {
        "current_target_word": current_word["word"],
        "exercise_type": current_word.get("type", "word"),
        "history": history,
    }
    result = coach_turn(state, None, None)
    reply = result["spoken_reply"]
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
        # view-state machinery
        "home",                                # view_state
        gr.update(visible=True),               # home_view
        gr.update(visible=False),              # practice_view
        # home UI
        welcome_html,                          # welcome_display
        resume_html,                           # resume_display
        gr.update(visible=has_resume),         # resume_btn
        # restored state for resume
        saved_level, saved_xp, saved_streak, saved_best, saved_hist,
        # username we remember through the session
        username or "",
    )


def _start_session(level: int, history: list, xp: int, streak: int, best_streak: int):
    """Shared bootstrap when entering the practice view at a given level."""
    first = get_next_word(history, "", explicit_level=level)
    reply, intro_audio, messages = _intro_for_word(first, history, [])
    return (
        # view-state
        "practice",
        gr.update(visible=False),              # home_view
        gr.update(visible=True),               # practice_view
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


def on_pick_level(level: int, profile: gr.OAuthProfile | None):
    """User tapped one of the five level buttons. Keep accumulated XP,
    best_streak, and history — just jump to the chosen level."""
    saved = _load_session_state(profile)
    return _start_session(
        level=level,
        history=saved["history"],
        xp=saved["xp"],
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
        "home",
        gr.update(visible=True),               # home_view
        gr.update(visible=False),              # practice_view
        resume_html,                           # resume_display
        gr.update(visible=bool(resume_html)),  # resume_btn
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


def on_submit(audio_path, current_word, history, messages,
              level, xp, streak, best_streak, username,
              profile: gr.OAuthProfile | None):
    if not current_word or not audio_path:
        return (current_word, history, messages, level, xp, streak, best_streak,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update())

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
                _level_bar_html(level, xp, streak), gr.update())

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
with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="Rivet R Coach") as demo:

    # ---- Persistent state ----
    view_state           = gr.State("home")
    word_state           = gr.State()
    history_state        = gr.State([])
    transcript_state     = gr.State([])
    level_state          = gr.State(0)
    xp_state             = gr.State(0)
    streak_state         = gr.State(0)
    best_streak_state    = gr.State(0)
    remembered_user_state = gr.State("")

    # ---- Hidden JS bridges ----
    state_bridge   = gr.Textbox(value="idle",                          visible=False, elem_id="rivet-state-bridge")
    bubble_bridge  = gr.Textbox(value="Just a sec — getting ready...", visible=False, elem_id="rivet-bubble-bridge")
    correct_bridge = gr.Textbox(value="false",                         visible=False, elem_id="rivet-correct-bridge")
    ready_bridge   = gr.Textbox(value="false",                         visible=False, elem_id="rivet-ready-bridge")

    # ---- Header ----
    with gr.Row(elem_classes=["rivet-header-row"]):
        gr.HTML(
            "<div class='rivet-header'>"
            "  <span class='rivet-wordmark'>rivet</span>"
            "  <button class='rivet-about' onclick=\"window.openRivetAbout()\" "
            "    aria-label='About Rivet R Coach'>About</button>"
            "</div>"
        )
        login_btn = gr.LoginButton(
            value="Sign in with Hugging Face",
            elem_classes=["rivet-login-btn"],
        )

    # ---- About modal ----
    gr.HTML(
        "<div id='rivet-about-modal' class='rivet-modal-backdrop' role='dialog' aria-modal='true'>"
        "  <div class='rivet-modal'>"
        "    <h2>About rivet</h2>"
        f"   <p>{DISCLAIMER}</p>"
        "    <p>Phoneme scoring: wav2vec2 (facebook/wav2vec2-lv-60-espeak-cv-ft) + "
        "       Praat formant analysis with CTC-guided /r/ segment isolation.<br>"
        "       Coaching: NVIDIA Nemotron 3 Nano 4B via Hugging Face Inference Providers.<br>"
        "       Voice: Kyutai pocket-tts.</p>"
        "    <p>Sign in with Hugging Face to save your progress across sessions.</p>"
        "    <p>Built for the Build Small Hackathon, June 2026.</p>"
        "    <button class='rivet-modal-close' onclick=\"window.closeRivetAbout()\">Close</button>"
        "  </div>"
        "</div>"
    )

    # ============================================================
    # HOME view
    # ============================================================
    with gr.Column(visible=True, elem_classes=["rivet-home"]) as home_view:

        welcome_display = gr.HTML(elem_id="rivet-welcome-wrap")
        resume_display  = gr.HTML(elem_id="rivet-resume-wrap")
        resume_btn      = gr.Button(
            "▶ Resume where you left off",
            variant="primary",
            visible=False,
            elem_id="resume-btn",
            elem_classes=["rivet-resume-btn"],
        )

        gr.HTML("<div class='rivet-pick-title'>Choose a level to practice</div>")

        with gr.Row(elem_classes=["rivet-level-grid"]):
            level_btn_0 = gr.Button(
                "Level 1\nSyllables", variant="secondary",
                elem_id="level-btn-0", elem_classes=["rivet-level-btn"],
            )
            level_btn_1 = gr.Button(
                "Level 2\nStarting Words", variant="secondary",
                elem_id="level-btn-1", elem_classes=["rivet-level-btn"],
            )
            level_btn_2 = gr.Button(
                "Level 3\nMiddle & End", variant="secondary",
                elem_id="level-btn-2", elem_classes=["rivet-level-btn"],
            )
            level_btn_3 = gr.Button(
                "Level 4\nVocalic R", variant="secondary",
                elem_id="level-btn-3", elem_classes=["rivet-level-btn"],
            )
            level_btn_4 = gr.Button(
                "Level 5\nPhrases", variant="secondary",
                elem_id="level-btn-4", elem_classes=["rivet-level-btn"],
            )

        gr.HTML(f"<div class='rivet-footer'>{DISCLAIMER}</div>")

    # ============================================================
    # PRACTICE view
    # ============================================================
    with gr.Column(visible=False, elem_classes=["rivet-practice"]) as practice_view:

        with gr.Row(elem_classes=["rivet-practice-top"]):
            back_btn = gr.Button(
                "← Home",
                variant="secondary",
                elem_id="back-btn",
                elem_classes=["rivet-back-btn"],
            )

        # ---- Progress bar ----
        progress_display = gr.HTML(elem_id="rivet-progress-wrap")
        # Level-up notification
        levelup_display  = gr.HTML(elem_id="rivet-levelup-wrap")

        # ---- Main two-column layout ----
        with gr.Row(elem_classes=["rivet-main"]):

            # ── Left column: character + transcript ──
            with gr.Column(scale=2, elem_classes=["rivet-card"]):
                gr.HTML(CHARACTER_HTML)
                transcript = gr.Chatbot(
                    label="Session log",
                    type="messages",
                    height=280,
                    show_copy_button=False,
                )

            # ── Right column: word + controls ──
            with gr.Column(scale=3, elem_classes=["rivet-card"]):
                word_display = gr.HTML()
                ipa_display  = gr.HTML()

                wren_audio = gr.Audio(
                    label="Wren's voice",
                    interactive=False,
                    autoplay=True,
                    type="numpy",
                    elem_id="wren-audio",
                )

                hear_audio = gr.Audio(
                    label="Target pronunciation (click Hear it)",
                    interactive=False,
                    autoplay=True,
                    type="numpy",
                    elem_id="hear-audio",
                )

                with gr.Row(elem_classes=["rivet-btn-row"]):
                    hear_btn = gr.Button(
                        "🔊 Hear it",
                        variant="secondary",
                        elem_id="hear-btn",
                    )
                    next_btn = gr.Button(
                        "Next →",
                        variant="primary",
                        elem_id="next-btn",
                    )

                mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎙️ Record yourself",
                    elem_id="user-mic",
                )

        gr.HTML(f"<div class='rivet-footer'>{DISCLAIMER}</div>")

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
    ]

    next_outputs = [
        word_state, history_state, transcript_state,
        level_state, xp_state, streak_state, best_streak_state,
        word_display, ipa_display, progress_display,
        wren_audio, transcript, mic,
        bubble_bridge, correct_bridge, ready_bridge,
        levelup_display,
    ]

    # start_session writes these outputs in this order (matches _start_session)
    start_session_outputs = [
        view_state, home_view, practice_view,
        word_state, history_state, transcript_state,
        level_state, xp_state, streak_state, best_streak_state,
        word_display, ipa_display, progress_display,
        wren_audio, transcript, mic,
        bubble_bridge, correct_bridge, ready_bridge,
        levelup_display,
    ]

    app_load_outputs = [
        view_state, home_view, practice_view,
        welcome_display, resume_display, resume_btn,
        level_state, xp_state, streak_state, best_streak_state, history_state,
        remembered_user_state,
    ]

    back_home_outputs = [
        view_state, home_view, practice_view,
        resume_display, resume_btn,
    ]

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

    # App boot — render home, populate welcome + resume from progress dataset
    demo.load(on_app_load, inputs=None, outputs=app_load_outputs)

    # Level pickers — re-load progress from dataset at click time, then
    # jump to the chosen level. OAuthProfile is auto-injected by Gradio.
    def _wire_level_btn(btn, level):
        btn.click(
            on_pick_level,
            inputs=[gr.State(level)],
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

    # JavaScript glue: open/close About modal
    gr.HTML("""
    <script>
    (function initRivetModal() {
      const modal = document.getElementById('rivet-about-modal');
      if (!modal) { setTimeout(initRivetModal, 200); return; }
      window.openRivetAbout  = () => modal.classList.add('open');
      window.closeRivetAbout = () => modal.classList.remove('open');
      modal.addEventListener('click', e => {
        if (e.target === modal) window.closeRivetAbout();
      });
    })();
    </script>
    """)


if __name__ == "__main__":
    demo.launch()
