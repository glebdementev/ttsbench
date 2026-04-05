from pathlib import Path

import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .analysis.pipeline import run_pipeline
from .storage import (
    load_library, save_library, add_entry, get_entry,
    update_entry, remove_entry, get_audio_path,
)

app = FastAPI(title="TTSBench")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}


def _analyze_file(audio_path: Path) -> dict:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    return run_pipeline(y, sr)


@app.get("/api/audio")
async def list_audio():
    return load_library()


@app.post("/api/audio")
async def upload_audio(
    file: UploadFile = File(...),
    alias: str = Form(""),
):
    if not file.filename:
        raise HTTPException(400, "Файл не выбран")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Неподдерживаемый формат: {suffix}")

    audio_bytes = await file.read()
    entry_alias = alias.strip() or Path(file.filename).stem

    entry = add_entry(entry_alias, file.filename, audio_bytes, suffix)

    try:
        audio_path = get_audio_path(entry["id"])
        analysis = _analyze_file(audio_path)
        entry = update_entry(entry["id"], {"analysis": analysis})
    except Exception as e:
        remove_entry(entry["id"])
        raise HTTPException(422, f"Ошибка обработки аудио: {e}")

    return entry


@app.delete("/api/audio/{entry_id}")
async def delete_audio(entry_id: str):
    if not remove_entry(entry_id):
        raise HTTPException(404, "Запись не найдена")
    return {"ok": True}


@app.patch("/api/audio/{entry_id}")
async def patch_audio(entry_id: str, body: dict):
    alias = body.get("alias", "").strip()
    if not alias:
        raise HTTPException(400, "Пустой алиас")
    entry = update_entry(entry_id, {"alias": alias})
    if not entry:
        raise HTTPException(404, "Запись не найдена")
    return entry


@app.post("/api/reanalyze")
async def reanalyze_all():
    lib = load_library()
    errors = []

    for entry in lib:
        audio_path = get_audio_path(entry["id"])
        if not audio_path:
            errors.append(f"{entry['alias']}: файл не найден")
            continue
        try:
            entry["analysis"] = _analyze_file(audio_path)
        except Exception as e:
            errors.append(f"{entry['alias']}: {e}")

    save_library(lib)
    return {"entries": lib, "errors": errors}


@app.get("/api/audio/{entry_id}/file")
async def serve_audio(entry_id: str):
    audio_path = get_audio_path(entry_id)
    if not audio_path:
        raise HTTPException(404, "Файл не найден")
    return FileResponse(audio_path)


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{filename:path}")
async def static_files(filename: str):
    if filename.startswith("api/"):
        raise HTTPException(404)
    file_path = (FRONTEND_DIR / filename).resolve()
    if file_path.is_file() and str(file_path).startswith(str(FRONTEND_DIR)):
        return FileResponse(file_path)
    raise HTTPException(404)
