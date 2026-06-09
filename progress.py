"""Per-user progress persistence backed by a Hugging Face Dataset.

Layout in the dataset repo:
    progress/<username>.json

Each JSON document looks like:
    {
        "username":    "ishaan",
        "level":       2,
        "xp":          145,
        "streak":      3,
        "best_streak": 11,
        "history":     [<last 50 attempts>],
        "updated_at":  "2026-06-08T19:42:11Z"
    }

Failure modes are silent — if HF_TOKEN is missing, or the dataset
doesn't exist, or the network is flaky, load_progress returns the
empty default and save_progress is a no-op. The practice loop keeps
working; the user just loses cross-device sync.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

DATASET_REPO = os.environ.get("PROGRESS_DATASET", "IndianChess/rivet-progress")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HISTORY_CAP  = 50   # cap stored history to keep files small


def _empty(username: str = "") -> dict:
    return {
        "username":    username,
        "level":       0,
        "xp":          0,
        "streak":      0,
        "best_streak": 0,
        "history":     [],
        "updated_at":  None,
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _user_path(username: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in username)
    return f"progress/{safe}.json"


def _api():
    """Lazy huggingface_hub HfApi handle. Returns None if no token."""
    if not HF_TOKEN:
        return None
    try:
        from huggingface_hub import HfApi
        return HfApi(token=HF_TOKEN)
    except Exception as e:  # noqa: BLE001
        print(f"[progress] huggingface_hub unavailable: {e}")
        return None


def load_progress(username: str | None) -> dict:
    """Fetch saved progress for a signed-in user.

    Returns the empty default if the user has no entry yet, or if the
    dataset can't be reached. Never raises.
    """
    if not username:
        return _empty("")

    api = _api()
    if api is None:
        return _empty(username)

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=_user_path(username),
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty(username)
        # Re-stamp username in case it was renamed / shared file
        data["username"] = username
        return {**_empty(username), **data}
    except Exception as e:  # noqa: BLE001  - load failures are non-fatal
        # Most common: 404 (no entry yet). Don't spam logs on those.
        msg = str(e)
        if "404" not in msg and "EntryNotFound" not in msg:
            print(f"[progress] load failed for {username}: {e}")
        return _empty(username)


def save_progress(username: str | None, payload: dict) -> bool:
    """Write the user's progress back to the dataset.

    Returns True on success, False if persistence is unavailable or the
    write failed. Never raises.
    """
    if not username:
        return False

    api = _api()
    if api is None:
        return False

    doc = {**_empty(username), **payload, "username": username, "updated_at": _now()}
    history = doc.get("history") or []
    if isinstance(history, list) and len(history) > HISTORY_CAP:
        doc["history"] = history[-HISTORY_CAP:]

    try:
        from huggingface_hub import CommitOperationAdd
        body = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
        api.create_commit(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(
                    path_in_repo=_user_path(username),
                    path_or_fileobj=body,
                )
            ],
            commit_message=f"progress: update {username} @ {int(time.time())}",
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[progress] save failed for {username}: {e}")
        return False
