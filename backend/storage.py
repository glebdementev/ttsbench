import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio"
LIBRARY_FILE = DATA_DIR / "library.json"


def _ensure_dirs():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def load_library() -> list[dict]:
    _ensure_dirs()
    if not LIBRARY_FILE.exists():
        return []
    return json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))


class _SafeEncoder(json.JSONEncoder):
    def default(self, o):
        return super().default(o)

    def encode(self, o):
        return super().encode(_sanitize(o))


def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def save_library(entries: list[dict]):
    _ensure_dirs()
    LIBRARY_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2, cls=_SafeEncoder),
        encoding="utf-8",
    )


def add_entry(alias: str, original_filename: str, audio_bytes: bytes, suffix: str) -> dict:
    _ensure_dirs()
    entry_id = uuid.uuid4().hex[:8]
    audio_path = AUDIO_DIR / f"{entry_id}{suffix}"
    audio_path.write_bytes(audio_bytes)

    entry = {
        "id": entry_id,
        "alias": alias,
        "filename": original_filename,
        "suffix": suffix,
        "added_at": datetime.now(timezone.utc).isoformat(),
        "analysis": None,
    }

    lib = load_library()
    lib.append(entry)
    save_library(lib)
    return entry


def get_entry(entry_id: str) -> dict | None:
    for e in load_library():
        if e["id"] == entry_id:
            return e
    return None


def update_entry(entry_id: str, updates: dict) -> dict | None:
    lib = load_library()
    for e in lib:
        if e["id"] == entry_id:
            e.update(updates)
            save_library(lib)
            return e
    return None


def remove_entry(entry_id: str) -> bool:
    lib = load_library()
    new_lib = [e for e in lib if e["id"] != entry_id]
    if len(new_lib) == len(lib):
        return False

    removed = next(e for e in lib if e["id"] == entry_id)
    audio_path = AUDIO_DIR / f"{entry_id}{removed.get('suffix', '.wav')}"
    audio_path.unlink(missing_ok=True)

    save_library(new_lib)
    return True


def get_audio_path(entry_id: str) -> Path | None:
    entry = get_entry(entry_id)
    if not entry:
        return None
    p = AUDIO_DIR / f"{entry_id}{entry.get('suffix', '.wav')}"
    return p if p.exists() else None
