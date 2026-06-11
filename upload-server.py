#!/usr/bin/env python3
"""
upload-server.py — FastAPI сервер транскрибации видео → текст.

Файловая очередь (работает с несколькими Uvicorn worker'ами):
  queue/pending/<id>.json   — задача ожидает
  queue/active/<id>.json    — задача в обработке
  queue/done/<id>.json      — задача выполнена

Запуск:
    uvicorn upload-server:app --host 0.0.0.0 --port 8080 --workers 2
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import urllib.parse

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# ─── Конфиг ────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
QUEUE_DIR = PROJECT_DIR / "queue"
PENDING_DIR = QUEUE_DIR / "pending"
ACTIVE_DIR = QUEUE_DIR / "active"
DONE_DIR = QUEUE_DIR / "done"

PORT = int(os.environ.get("PORT", 8080))
HOST = os.environ.get("HOST", "0.0.0.0")
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 ГБ

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PENDING_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
DONE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Basic Auth ─────────────────────────────────────────────────────

BASIC_AUTH_USER = os.environ.get("BASIC_AUTH_USER", "admin")
BASIC_AUTH_PASS = os.environ.get("BASIC_AUTH_PASS", "admin123")

security = HTTPBasic(auto_error=False)


def require_auth(credentials: HTTPBasicCredentials | None = Depends(security)):
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Basic"},
        )
    if credentials.username != BASIC_AUTH_USER or credentials.password != BASIC_AUTH_PASS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

# ─── Конфиг ────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
QUEUE_DIR = PROJECT_DIR / "queue"
PENDING_DIR = QUEUE_DIR / "pending"
ACTIVE_DIR = QUEUE_DIR / "active"
DONE_DIR = QUEUE_DIR / "done"

PORT = int(os.environ.get("PORT", 8080))
HOST = os.environ.get("HOST", "0.0.0.0")
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 ГБ

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PENDING_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
DONE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="🎬 Video → Text Transcriber",
    version="3.1",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Файловая очередь ───────────────────────────────────────────────

def _job_id() -> str:
    return uuid.uuid4().hex[:12]

def _read_job(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _write_job(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)

def _list_jobs(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        return []
    return sorted(dir_path.iterdir())

def enqueue(filename: str) -> int:
    """Добавляет задачу в очередь. Возвращает позицию."""
    jid = _job_id()
    data = {
        "id": jid,
        "filename": filename,
        "status": "pending",
        "created": time.time(),
        "started": None,
    }
    _write_job(PENDING_DIR / f"{jid}.json", data)
    # Позиция = количество задач в pending + active
    return len(_list_jobs(PENDING_DIR)) + len(_list_jobs(ACTIVE_DIR))

def claim_one() -> Optional[str]:
    """Перемещает одну задачу из pending → active. Возвращает filename или None."""
    for f in _list_jobs(PENDING_DIR):
        data = _read_job(f)
        if data is None:
            continue
        data["status"] = "active"
        data["started"] = time.time()
        dest = ACTIVE_DIR / f.name
        # Атомарный перенос через файловую систему
        try:
            _write_job(dest, data)
            f.unlink(missing_ok=True)
            return data["filename"]
        except Exception:
            continue
    return None

def finish_job(filename: str, success: bool, error_msg: str = ""):
    """Перемещает active → done с результатом."""
    for f in _list_jobs(ACTIVE_DIR):
        data = _read_job(f)
        if data and data.get("filename") == filename:
            data["status"] = "done" if success else "error"
            data["error"] = error_msg
            data["finished"] = time.time()
            dest = DONE_DIR / f.name
            _write_job(dest, data)
            f.unlink(missing_ok=True)
            return

def get_job_status(filename: str) -> Optional[dict]:
    """Ищет задачу по имени файла во всех статусах."""
    for d, status in [(PENDING_DIR, "queued"), (ACTIVE_DIR, "active"), (DONE_DIR, "done")]:
        for f in _list_jobs(d):
            data = _read_job(f)
            if data and data.get("filename") == filename:
                data["status"] = data.get("status", status)

                # Считаем позицию в очереди
                if data["status"] == "queued":
                    pending = _list_jobs(PENDING_DIR)
                    position = 1
                    for pf in pending:
                        pd = _read_job(pf)
                        if pd and pd.get("id") == data.get("id"):
                            break
                        position += 1
                    data["position"] = position

                return data
    return None

def queue_stats() -> dict:
    """Статистика очереди."""
    pending = len(_list_jobs(PENDING_DIR))
    active = len(_list_jobs(ACTIVE_DIR))
    done_d = len(_list_jobs(DONE_DIR))
    return {"pending": pending, "active": active, "done": done_d, "total": pending + active + done_d}


# ─── Подхват файлов из input/ ──────────────────────────────────────

def pickup_input_files():
    """Сканирует input/ и ставит новые файлы в очередь.
    Чистит stale active/ и error done/ задачи (воркер мог упасть с OOM)."""
    count = 0

    # Stale active → возвращаем в pending + сразу запускаем обработку
    for f in _list_jobs(ACTIVE_DIR):
        data = _read_job(f)
        if data:
            fname = data.get("filename", "")
            # Если файл есть в input — оставляем в active и сразу запускаем
            if (INPUT_DIR / fname).exists() and not (OUTPUT_DIR / f"{Path(fname).stem}.txt").exists():
                print(f"🔄  Stale active, запускаю: {fname}")
                asyncio.create_task(process_file(fname))
                count += 1
                continue
            # Иначе — в pending
            data["status"] = "pending"
            data["started"] = None
            data.pop("error", None)
            dest = PENDING_DIR / f.name
            _write_job(dest, data)
            f.unlink(missing_ok=True)
            print(f"🔄  Stale active → pending: {fname}")
            count += 1

    # Error done → удаляем маркер (файл перейдёт в pending ниже)
    for f in _list_jobs(DONE_DIR):
        data = _read_job(f)
        if data and data.get("status") == "error":
            fname = data.get("filename", "")
            f.unlink(missing_ok=True)
            print(f"🗑️  Removed error marker: {fname}")

    for f in sorted(INPUT_DIR.iterdir()):
        if not f.is_file():
            continue
        stem = f.stem
        # Пропускаем, если уже в очереди или готов
        if get_job_status(f.name) is not None:
            continue
        if (OUTPUT_DIR / f"{stem}.txt").exists():
            continue
        enqueue(f.name)
        print(f"↩  Pickup from input: {f.name}")
        count += 1
    return count


# ─── Фоновый воркер ────────────────────────────────────────────────

async def process_file(filename: str):
    """Запускает транскрибацию в подпроцессе."""
    stem = Path(filename).stem
    progress_file = OUTPUT_DIR / f"{stem}.progress.json"
    input_path = INPUT_DIR / filename
    output_txt = OUTPUT_DIR / f"{stem}.txt"

    if not input_path.exists():
        finish_job(filename, False, "Файл не найден в input/")
        return

    print(f"\n🎯  Старт: {filename}")

    cmd = [
        "python", "transcribe.py",
        "--model", os.environ.get("WHISPER_MODEL", "base"),
        "--lang", os.environ.get("WHISPER_LANG", "ru"),
        "--device", os.environ.get("WHISPER_DEVICE", "cpu"),
        "--compute", os.environ.get("WHISPER_COMPUTE", "int8"),
        "--progress-file", str(progress_file),
        str(input_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=86400)

        if proc.returncode == 0:
            if output_txt.exists():
                finish_job(filename, success=True)
                print(f"✅  Готово: {filename}")
                # Чистим input
                try:
                    input_path.unlink(missing_ok=True)
                except Exception:
                    pass
                # Чистим progress
                try:
                    progress_file.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                err = "Файл результата не найден"
                finish_job(filename, False, err)
                print(f"❌  {err}: {filename}")
        else:
            err_text = stderr.decode("utf-8", errors="replace")[-500:] if stderr else ""
            if not err_text:
                err_text = stdout.decode("utf-8", errors="replace")[-500:] if stdout else "Неизвестная ошибка"
            finish_job(filename, False, err_text)
            print(f"❌  Ошибка ({proc.returncode}): {err_text[:200]}")

    except asyncio.TimeoutError:
        finish_job(filename, False, "Таймаут (>24ч)")
        print(f"❌  Таймаут: {filename}")
    except Exception as e:
        finish_job(filename, False, str(e))
        print(f"❌  {e}")


async def worker_loop():
    """Фоновый цикл: забирает задачи из очереди и обрабатывает."""
    while True:
        filename = claim_one()
        if filename:
            await process_file(filename)
        else:
            await asyncio.sleep(2)


@app.on_event("startup")
async def startup():
    # Подхватываем файлы из input/ + чистим stale
    n = pickup_input_files()
    if n:
        print(f"📦  Подхвачено из input/: {n} файл(ов)")

    # Стартуем фоновый воркер — оба воркера крутят цикл,
    # но claim_one() атомарен (temp+rename), только один возьмёт задачу
    asyncio.create_task(worker_loop())

    print(f"🎬  Сервер запущен (PID {os.getpid()}): http://localhost:{PORT}")


# ─── Вспомогательные функции ────────────────────────────────────────

def get_stem(filename: str) -> str:
    return Path(filename).stem

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
    ".ts", ".flv", ".wmv", ".3gp", ".mpg", ".mpeg",
    ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac",
    ".wma", ".opus", ".aiff",
}


# ─── Служебные ручки ──────────────────────────────────────────────

@app.get("/health")
async def health(_auth=Depends(require_auth)):
    stats = queue_stats()
    return {"status": "ok", "queue": stats}


@app.get("/files")
async def list_files(_auth=Depends(require_auth)):
    """Список готовых .txt файлов для скачивания."""
    files = []
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.suffix == ".txt" and not p.name.endswith(".partial.txt"):
            size_kb = round(p.stat().st_size / 1024, 1)
            mtime = p.stat().st_mtime
            files.append({
                "name": p.name,
                "size_kb": size_kb,
                "modified": mtime,
            })
    return {"files": files}


@app.get("/uploaded")
async def list_uploaded(_auth=Depends(require_auth)):
    """Список загруженных видео (файлы в input/)."""
    files = []
    for p in sorted(INPUT_DIR.iterdir()):
        if p.is_file():
            size_mb = round(p.stat().st_size / (1024 * 1024), 1)
            mtime = p.stat().st_mtime
            # Статус в очереди
            qs = ""
            for d, label in [(PENDING_DIR, "⏳"), (ACTIVE_DIR, "⚙️"), (DONE_DIR, "✅")]:
                for f in _list_jobs(d):
                    data = _read_job(f)
                    if data and data.get("filename") == p.name:
                        qs = label
            files.append({
                "name": p.name,
                "size_mb": size_mb,
                "modified": mtime,
                "queue_status": qs,
            })
    return {"files": files}


# ─── Главная HTML ──────────────────────────────────────────────────

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🎬 Видео → Текст</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:system-ui,-apple-system,sans-serif;background:#f1f5f9;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
  .card{background:#fff;border-radius:24px;box-shadow:0 20px 35px -8px rgba(0,0,0,.15);padding:40px;width:600px;max-width:calc(100vw - 40px)}
  h1{font-size:24px;margin-bottom:4px;color:#1e293b}
  .sub{color:#64748b;font-size:14px;margin-bottom:24px}
  .dropzone{border:2px dashed #cbd5e1;border-radius:16px;padding:48px 24px;text-align:center;cursor:pointer;transition:.2s;background:#f8fafc}
  .dropzone.dragover{border-color:#6366f1;background:#eef2ff}
  .dropzone.has-file{border-color:#22c55e;background:#f0fdf4;border-style:solid}
  .dropzone-icon{font-size:48px;margin-bottom:8px;color:#94a3b8}
  .dropzone-text{color:#64748b;font-size:15px}
  .dropzone-text strong{color:#6366f1}
  .dropzone-hint{font-size:12px;color:#94a3b8;margin-top:4px}
  input[type=file]{display:none}

  .section{display:none;margin-top:20px}
  .section.show{display:block}

  .file-card{padding:14px 16px;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0}
  .file-card .name{font-weight:600;color:#1e293b;word-break:break-all;margin-bottom:2px}
  .file-card .size{font-size:13px;color:#64748b}

  .progress-bar{height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;margin-top:16px}
  .progress-bar .fill{height:100%;width:0%;background:linear-gradient(90deg,#6366f1,#8b5cf6);border-radius:4px;transition:width .5s}

  .status-msg{padding:12px 16px;border-radius:12px;font-size:14px;margin-top:12px;line-height:1.5}
  .status-msg.idle{display:flex;align-items:center;gap:10px;background:#f8fafc;color:#64748b;border:1px solid #e2e8f0}
  .status-msg.queued{background:#fffbeb;color:#d97706;border:1px solid #fde68a}
  .status-msg.processing{background:#eef2ff;color:#4338ca;border:1px solid #c7d2fe}
  .status-msg.error{background:#fef2f2;color:#dc2626;border:1px solid #fecaca}
  .status-msg.done{background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0}

  .spinner{display:inline-block;width:16px;height:16px;border:2px solid #c7d2fe;border-top-color:#4338ca;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:4px}
  @keyframes spin{to{transform:rotate(360deg)}}

  .btn{display:inline-flex;align-items:center;gap:8px;padding:12px 24px;border:none;border-radius:12px;font-size:15px;cursor:pointer;transition:.2s;text-decoration:none;font-family:inherit}
  .btn-primary{background:#6366f1;color:#fff}
  .btn-primary:hover{background:#4f46e5}
  .btn-primary:disabled{background:#cbd5e1;cursor:not-allowed}

  .flex{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap}

  .preview-box{margin-top:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:0;overflow:hidden}
  .preview-box .preview-header{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;background:#f1f5f9;border-bottom:1px solid #e2e8f0;font-size:13px;font-weight:600;color:#475569}
  .preview-box .preview-body{padding:16px;max-height:220px;overflow-y:auto;font-size:14px;line-height:1.6;color:#334155;white-space:pre-wrap;word-break:break-word}
  .preview-box .preview-body:empty::after{content:"Пока нет расшифровки…";color:#94a3b8;font-style:italic}

  .eta{font-size:13px;color:#64748b;margin-top:4px}
  .seg-count{font-size:12px;color:#94a3b8}

  .queue-badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600;background:#f1f5f9;color:#475569;margin-left:8px}
  .queue-badge.active{background:#eef2ff;color:#4338ca}

  .files-list{margin-top:20px}
  .files-list details{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden}
  .files-list details summary{padding:12px 16px;cursor:pointer;font-weight:600;font-size:14px;color:#1e293b;user-select:none;display:flex;align-items:center;gap:8px}
  .files-list details summary::-webkit-details-marker{color:#6366f1}
  .files-list .file-row{display:flex;align-items:center;justify-content:space-between;padding:10px 16px 10px 32px;border-top:1px solid #f1f5f9;transition:.15s}
  .files-list .file-row:hover{background:#f1f5f9}
  .files-list .file-row .fname{font-size:14px;color:#1e293b;word-break:break-all;flex:1}
  .files-list .file-row .finfo{font-size:12px;color:#94a3b8;margin:0 12px;white-space:nowrap}
  .files-list .file-row .fdl{font-size:13px;color:#6366f1;text-decoration:none;font-weight:600;white-space:nowrap}
  .files-list .file-row .fdl:hover{color:#4f46e5;text-decoration:underline}
  .files-list .file-row .fstatus{font-size:11px;color:#64748b;white-space:nowrap;margin-right:8px}
  .btn-del{background:none;border:none;cursor:pointer;font-size:14px;padding:4px;opacity:.6;transition:.15s;line-height:1}
  .btn-del:hover{opacity:1}
  .files-empty{text-align:center;padding:20px;color:#94a3b8;font-size:13px}
</style>
</head>
<body>
<div class="card">
  <h1>🎬 Видео → Текст</h1>
  <p class="sub">Перетащи видео — получишь расшифровку</p>

  <div class="dropzone" id="dropzone">
    <div class="dropzone-icon">📁</div>
    <div class="dropzone-text"><strong>Выберите файл</strong> или перетащите</div>
    <div class="dropzone-hint">MP4, MKV, MOV, AVI, WAV, MP3 · до 10 ГБ</div>
  </div>
  <input type="file" id="fileInput" accept="video/*,audio/*">

  <!-- Список загруженных видео -->
  <div class="files-list" style="margin-top:12px">
    <details>
      <summary>🎬 Загруженные видео <span class="queue-badge" id="uploadedCount">0</span></summary>
      <div id="uploadedBody">
        <div class="files-empty">Нет загруженных файлов</div>
      </div>
    </details>
  </div>

  <!-- Список готовых файлов -->
  <div class="files-list" id="filesList">
    <details>
      <summary>📂 Готовые файлы <span class="queue-badge" id="filesCount">0</span></summary>
      <div id="filesBody">
        <div class="files-empty">Пока нет готовых расшифровок</div>
      </div>
    </details>
  </div>

  <!-- Прогресс загрузки -->
  <div class="progress-bar" id="uploadProgress" style="display:none">
    <div class="fill" id="uploadFill"></div>
  </div>

  <!-- Инфо о файле -->
  <div class="section" id="fileSection">
    <div class="file-card">
      <div class="name" id="fileName">—</div>
      <div class="size" id="fileSize"></div>
    </div>
  </div>

  <!-- Позиция в очереди -->
  <div class="status-msg queued" id="queueMsg" style="display:none">
    <span>⏳</span>
    <span id="queueText">В очереди…</span>
  </div>

  <!-- Статус обработки -->
  <div class="status-msg idle" id="statusMsg">
    <span>💡</span>
    <span>Загрузи видео, чтобы начать</span>
  </div>

  <!-- Превью частичного текста -->
  <div class="section" id="previewSection">
    <div class="preview-box">
      <div class="preview-header">
        <span>📝 Предпросмотр расшифровки</span>
        <span id="previewInfo" class="seg-count"></span>
      </div>
      <div class="preview-body" id="previewBody">Пока нет расшифровки…</div>
    </div>
  </div>

  <!-- Кнопки скачивания -->
  <div id="actions"></div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const uploadProgress = document.getElementById('uploadProgress');
const uploadFill = document.getElementById('uploadFill');
const statusMsg = document.getElementById('statusMsg');
const queueMsg = document.getElementById('queueMsg');
const queueText = document.getElementById('queueText');
const actions = document.getElementById('actions');
const fileSection = document.getElementById('fileSection');

// Basic Auth — добавляем заголовок ко всем fetch
const AUTH = 'Basic ' + btoa('admin:admin123');
const authHeaders = { 'Authorization': AUTH };

function authFetch(url, opts) {
  opts = opts || {};
  opts.headers = opts.headers || {};
  Object.assign(opts.headers, authHeaders);
  return fetch(url, opts);
}

let currentFilename = null;
let pollTimer = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => { e.preventDefault(); dropzone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFiles(fileInput.files); });

async function handleFiles(files) {
  const file = files[0];
  if (!file) return;

  fileName.textContent = file.name;
  fileSize.textContent = formatSize(file.size);
  fileSection.classList.add('show');
  queueMsg.style.display = 'none';
  setStatus('idle', '⏳ Загружаю...');
  actions.innerHTML = '';
  document.getElementById('previewSection').classList.remove('show');

  if (file.size > 10 * 1024 * 1024 * 1024) {
    setStatus('error', '❌ Файл слишком большой. Максимум 10 ГБ.');
    return;
  }

  uploadProgress.style.display = 'block';
  uploadFill.style.width = '0%';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) uploadFill.style.width = (e.loaded / e.total * 100) + '%';
    };
    const result = await new Promise((resolve, reject) => {
      xhr.onload = () => {
        if (xhr.status === 200) resolve(JSON.parse(xhr.responseText));
        else reject(new Error(xhr.responseText || 'Ошибка'));
      };
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.open('POST', '/upload');
      xhr.setRequestHeader('Authorization', AUTH);
      xhr.send(formData);
    });

    uploadFill.style.width = '100%';
    currentFilename = result.name;

    if (result.position > 1) {
      queueMsg.style.display = 'block';
      queueText.textContent = 'Позиция в очереди: ' + result.position;
      setStatus('idle', '⏳ В очереди на обработку…');
    } else {
      dropzone.classList.add('has-file');
      setStatus('processing', '<span class="spinner"></span> Начинаю обработку…');
    }

    startPolling(result.name);

  } catch (err) {
    uploadProgress.style.display = 'none';
    setStatus('error', '❌ ' + err.message);
  }
}

function startPolling(filename) {
  if (pollTimer) clearInterval(pollTimer);
  let lastPartialRefresh = 0;

  pollTimer = setInterval(async () => {
    try {
      const resp = await authFetch('/status?file=' + encodeURIComponent(filename));
      const data = await resp.json();

      if (data.status === 'queued') {
        const pos = data.position || 0;
        queueMsg.style.display = 'block';
        queueText.textContent = pos > 1
          ? 'Позиция в очереди: ' + pos
          : '⏳ Скоро начнётся…';
      } else {
        queueMsg.style.display = 'none';
      }

      if (data.status === 'processing') {
        let statusHtml;

        if (data.progress_pct > 0) {
          uploadFill.style.width = Math.min(data.progress_pct, 100) + '%';

          const pct = Math.round(data.progress_pct);
          const elapsed = data.elapsed_sec || 0;
          const elapsedStr = elapsed >= 3600
            ? (elapsed / 3600).toFixed(1) + ' ч'
            : Math.round(elapsed / 60) + ' мин';

          statusHtml = `<span class="spinner"></span> ${pct}% обработано · ${elapsedStr}`;

          if (data.segments_count) {
            statusHtml += `<br><span style="font-size:12px;color:#6366f1">${data.segments_count} фрагментов · «${escHtml((data.last_text || '').slice(0, 60))}»</span>`;
          }

          const now = Date.now();
          if (data.has_partial && now - lastPartialRefresh > 5000) {
            lastPartialRefresh = now;
            loadPartialPreview(filename);
          }
        } else {
          const elapsed = data.elapsed_sec || 0;
          const sec = Math.round(elapsed);
          statusHtml = `<span class="spinner"></span> Загрузка модели… (${sec} сек)`;
        }

        setStatus('processing', statusHtml);

      } else if (data.status === 'done') {
        clearInterval(pollTimer);
        pollTimer = null;
        uploadProgress.style.display = 'none';
        setStatus('done', '✅ Готово!');
        document.getElementById('previewSection').classList.remove('show');
        showDownload(filename);
        loadFilesList();
      } else if (data.status === 'error') {
        clearInterval(pollTimer);
        pollTimer = null;
        uploadProgress.style.display = 'none';
        document.getElementById('previewSection').classList.remove('show');
        setStatus('error', '❌ ' + (data.error || 'Ошибка'));
      }
    } catch (err) {
      // ignore polling errors
    }
  }, 2000);
}

function loadPartialPreview(filename) {
  const section = document.getElementById('previewSection');
  const body = document.getElementById('previewBody');
  const info = document.getElementById('previewInfo');

  authFetch('/preview?file=' + encodeURIComponent(filename))
    .then(r => {
      if (!r.ok) throw new Error('no preview');
      return r.text();
    })
    .then(text => {
      section.classList.add('show');
      if (text.length > 3000) {
        body.textContent = text.slice(0, 3000) + '\n\n… (показано 3000 символов)';
      } else {
        body.textContent = text;
      }
      info.textContent = (text.length > 3000 ? '3 000' : text.length) + ' символов';
    })
    .catch(() => {
      if (section.classList.contains('show') && !body.textContent.trim()) {
        section.classList.remove('show');
      }
    });
}

function showDownload(filename) {
  const stem = filename.replace(/\.[^.]+$/, '');
  const encodedName = encodeURIComponent(stem + '.txt');
  actions.innerHTML = `
    <div class="flex">
      <a class="btn btn-primary" href="/download/${encodedName}" download>📄 Скачать текст (.txt)</a>
    </div>
  `;
}

function setStatus(type, html) {
  statusMsg.className = 'status-msg ' + type;
  statusMsg.innerHTML = html;
}

function escHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' Б';
  if (bytes < 1024*1024)  return (bytes/1024).toFixed(1) + ' КБ';
  if (bytes < 1024*1024*1024) return (bytes/1024/1024).toFixed(1) + ' МБ';
  return (bytes/1024/1024/1024).toFixed(2) + ' ГБ';
}

// ── Список готовых файлов ──

function loadFilesList() {
  authFetch('/files')
    .then(r => r.json())
    .then(data => {
      const body = document.getElementById('filesBody');
      const count = document.getElementById('filesCount');
      const files = data.files || [];

      count.textContent = files.length;

      if (!files.length) {
        body.innerHTML = '<div class="files-empty">Нет готовых файлов</div>';
        return;
      }

      body.innerHTML = files.map(f => {
        const date = new Date(f.modified * 1000);
        const dateStr = date.toLocaleDateString('ru-RU', {
          day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
        });
        const sizeStr = f.size_kb < 1024
          ? f.size_kb + ' КБ'
          : (f.size_kb / 1024).toFixed(1) + ' МБ';
        const encoded = encodeURIComponent(f.name);
        return `<div class="file-row">
          <span class="fname">📄 ${escHtml(f.name)}</span>
          <span class="finfo">${sizeStr} · ${dateStr}</span>
          <a class="fdl" href="/download/${encoded}" download>📥</a>
          <button class="btn-del" onclick="deleteFile('${encoded}', 'output')" title="Удалить">🗑️</button>
        </div>`;
      }).join('');
    })
    .catch(() => {});
}

// ── Удаление файлов ──

function deleteFile(encoded, source) {
  const name = decodeURIComponent(encoded);
  if (!confirm('Удалить «' + name + '»?')) return;

  authFetch('/delete/' + encoded, { method: 'DELETE' })
    .then(r => {
      if (!r.ok) throw new Error('Ошибка удаления');
      return r.json();
    })
    .then(data => {
      loadFilesList();
      loadUploadedFiles();
      // Если удалили активный файл — сбрасываем интерфейс
      if (currentFilename === name) {
        if (pollTimer) clearInterval(pollTimer);
        currentFilename = null;
        setStatus('idle', '💡 Файл удалён');
        actions.innerHTML = '';
        uploadProgress.style.display = 'none';
      }
      setStatus('idle', '🗑️ Файл удалён');
    })
    .catch(err => {
      alert('Ошибка: ' + err.message);
    });
}

// ── Список загруженных видео ──

function loadUploadedFiles() {
  authFetch('/uploaded')
    .then(r => r.json())
    .then(data => {
      const body = document.getElementById('uploadedBody');
      const count = document.getElementById('uploadedCount');
      const files = data.files || [];
      count.textContent = files.length;
      if (!files.length) {
        body.innerHTML = '<div class="files-empty">Нет загруженных файлов</div>';
        return;
      }
      body.innerHTML = files.map(f => {
        const date = new Date(f.modified * 1000);
        const dateStr = date.toLocaleDateString('ru-RU', {
          day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
        });
        const sizeStr = f.size_mb < 1
          ? Math.round(f.size_mb * 1024) + ' КБ'
          : f.size_mb.toFixed(1) + ' МБ';
        const encoded = encodeURIComponent(f.name);
        return `<div class="file-row">
          <span class="fname">🎬 ${escHtml(f.name)}</span>
          <span class="finfo">${sizeStr} · ${dateStr}</span>
          <span class="fstatus">${f.queue_status}</span>
          <button class="btn-del" onclick="deleteFile('${encoded}', 'input')" title="Удалить">🗑️</button>
        </div>`;
      }).join('');
    })
    .catch(() => {});
}

// ── Проверка активной задачи при загрузке страницы ──

function checkActiveJob() {
  authFetch('/status')
    .then(r => r.json())
    .then(data => {
      if (data.active_file) {
        currentFilename = data.active_file;
        fileName.textContent = currentFilename;
        fileSection.classList.add('show');

        if (data.status === 'queued') {
          setStatus('idle', '⏳ В очереди: ' + currentFilename);
          queueMsg.style.display = 'block';
          queueText.textContent = 'Позиция в очереди: ' + (data.position || 1);
        } else if (data.status === 'processing' || data.status === 'loading_model') {
          dropzone.classList.add('has-file');

          // Определяем примерный размер файла из имени
          authFetch('/files').then(r => r.json()).then(fd => {
            const match = fd.files.find(f => f.name.startsWith(currentFilename.replace(/\.[^.]+$/, '')));
            if (match) fileSize.textContent = match.size_kb >= 1024
              ? (match.size_kb/1024).toFixed(1) + ' МБ'
              : match.size_kb + ' КБ';
          }).catch(()=>{});

          // Восстанавливаем состояние из ответа напрямую
          if (data.progress_pct > 0) {
            const pct = Math.round(data.progress_pct);
            uploadFill.style.width = Math.min(pct, 100) + '%';
            uploadProgress.style.display = 'block';
            const elapsed = data.elapsed_sec || 0;
            const elapsedStr = elapsed >= 3600
              ? (elapsed/3600).toFixed(1) + ' ч'
              : Math.round(elapsed/60) + ' мин';
            let html = `<span class="spinner"></span> ${pct}% · ${elapsedStr}`;
            if (data.chunk) html += `<br><span style="font-size:12px;color:#6366f1">Чанк ${data.chunk}</span>`;
            if (data.segments_count) {
              html += `<br><span style="font-size:12px;color:#6366f1">${data.segments_count} фрагментов · «${escHtml((data.last_text||'').slice(0,60))}»</span>`;
            }
            setStatus('processing', html);
            if (data.has_partial) {
              loadPartialPreview(currentFilename);
            }
          } else {
            setStatus('processing', '<span class="spinner"></span> Загрузка модели… (' + (data.elapsed_sec||0) + ' сек)');
          }

          startPolling(currentFilename);
        } else if (data.status === 'done') {
          setStatus('done', '✅ Готово: ' + currentFilename);
          showDownload(currentFilename);
          loadFilesList();
        } else if (data.status === 'error') {
          setStatus('error', '❌ ' + (data.error || 'Ошибка'));
        }
      }
    })
    .catch(() => {});
}

// При загрузке страницы — проверяем активную задачу и список файлов
checkActiveJob();
loadFilesList();
loadUploadedFiles();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index(_auth=Depends(require_auth)):
    return INDEX_HTML


# ─── Upload ─────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), _auth=Depends(require_auth)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Формат {ext} не поддерживается",
        )

    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Файл слишком большой (макс. 10 ГБ)")

    # Сохраняем файл
    safe_name = os.path.basename(file.filename)
    dest = INPUT_DIR / safe_name

    counter = 1
    while dest.exists():
        stem = Path(safe_name).stem
        ext = Path(safe_name).suffix
        dest = INPUT_DIR / f"{stem}_{counter}{ext}"
        counter += 1

    # Пишем файл чанками (чтобы не грузить весь в память)
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)

    # Ставим в очередь (файловая очередь, не блокирует event loop)
    position = enqueue(dest.name)

    size_mb = round(dest.stat().st_size / (1024 * 1024), 1)
    print(f"📥  Загружен: {dest.name} ({size_mb} МБ), позиция: {position}")

    return {
        "ok": True,
        "name": dest.name,
        "size_mb": size_mb,
        "position": position,
    }


# ─── Status ─────────────────────────────────────────────────────────

@app.get("/status")
async def get_status(file: Optional[str] = None, _auth=Depends(require_auth)):
    """Статус обработки файла. Если file не указан — отдаёт активный."""
    if file:
        return _get_file_status(file)

    # Без file — отдаём текущий статус очереди
    active_files = _list_jobs(ACTIVE_DIR)
    if active_files:
        data = _read_job(active_files[0])
        if data:
            fname = data.get("filename", "")
            result = _get_file_status(fname)
            result["active_file"] = fname
            return result

    pending_files = _list_jobs(PENDING_DIR)
    if pending_files:
        data = _read_job(pending_files[0])
        if data:
            return {"status": "queued", "active_file": data.get("filename", ""), "position": 1}

    return {"status": "idle"}


def _get_file_status(file: str) -> dict:
    """Статус для конкретного файла."""
    stem = Path(file).stem
    job = get_job_status(file)
    if job is None:
        txt = OUTPUT_DIR / f"{stem}.txt"
        if txt.exists():
            return {"status": "done", "txt": f"{stem}.txt"}
        return {"status": "idle"}

    if job["status"] == "queued":
        return {
            "status": "queued",
            "position": job.get("position", 0),
            "filename": file,
        }

    if job["status"] == "active":
        elapsed = time.time() - (job.get("started") or time.time())
        progress_file = OUTPUT_DIR / f"{stem}.progress.json"

        progress_data = {}
        if progress_file.exists():
            try:
                progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        pct = progress_data.get("progress_pct", 0)
        seg_count = progress_data.get("segments_count", 0)
        last_text = progress_data.get("text", "")
        chunk = progress_data.get("chunk", "")
        total_dur = progress_data.get("total_duration_sec", 0)

        return {
            "status": "processing" if pct > 0 else "loading_model",
            "filename": file,
            "progress_pct": round(pct, 1),
            "segments_count": seg_count,
            "last_text": last_text[:200] if last_text else "",
            "elapsed_sec": round(elapsed, 1),
            "chunk": chunk,
            "total_duration_sec": round(total_dur, 1),
            "has_partial": (OUTPUT_DIR / f"{stem}.partial.txt").exists(),
        }

    if job["status"] == "done":
        size_kb = 0
        txt_p = OUTPUT_DIR / f"{stem}.txt"
        if txt_p.exists():
            size_kb = round(txt_p.stat().st_size / 1024, 1)
        return {"status": "done", "filename": file, "txt": f"{stem}.txt", "size_kb": size_kb}

    if job.get("status") in ("error",):
        return {
            "status": "error",
            "filename": file,
            "error": job.get("error", "Неизвестная ошибка"),
        }

    return {"status": "idle"}


# ─── Preview ───────────────────────────────────────────────────────

@app.get("/preview")
async def get_preview(file: str, _auth=Depends(require_auth)):
    """Частичный текст в процессе обработки."""
    stem = Path(file).stem
    partial_path = OUTPUT_DIR / f"{stem}.partial.txt"
    if not partial_path.exists():
        raise HTTPException(status_code=404, detail="partial text not available yet")
    try:
        text = partial_path.read_text(encoding="utf-8")
        return PlainTextResponse(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Download ───────────────────────────────────────────────────────

@app.get("/download/{filename:path}")
async def download_file(filename: str, _auth=Depends(require_auth)):
    """Скачать готовый файл."""
    filepath = (OUTPUT_DIR / filename).resolve()
    if not str(filepath).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(status_code=403, detail="forbidden")

    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    mime_map = {".txt": "text/plain; charset=utf-8"}
    content_type = mime_map.get(filepath.suffix, "application/octet-stream")

    data = filepath.read_bytes()
    safe_name = filepath.name.encode("ascii", errors="replace").decode("ascii")

    # RFC 5987 — поддержка кириллицы в имени файла
    try:
        filepath.name.encode("ascii")
        disposition = f'attachment; filename="{filepath.name}"'
    except UnicodeEncodeError:
        disposition = (
            f'attachment; filename="{safe_name}"; '
            f"filename*=UTF-8''{urllib.parse.quote(filepath.name)}"
        )

    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Content-Disposition": disposition,
            "Content-Length": str(len(data)),
        },
    )


# ─── Delete ─────────────────────────────────────────────────────────


@app.delete("/delete/{filename:path}")
async def delete_file(filename: str, _auth=Depends(require_auth)):
    """Удаляет файл из input/ или output/."""
    stem = Path(filename).stem

    # Удаляем из input/
    input_file = (INPUT_DIR / filename).resolve()
    if str(input_file).startswith(str(INPUT_DIR.resolve())) and input_file.exists():
        input_file.unlink()
        # Удаляем маркер очереди
        for d in [PENDING_DIR, ACTIVE_DIR, DONE_DIR]:
            for f in _list_jobs(d):
                data = _read_job(f)
                if data and data.get("filename") == filename:
                    f.unlink(missing_ok=True)
        print(f"🗑️  Удалён input: {filename}")

    # Удаляем из output/
    deleted_output = []
    for suffix in [".txt", ".partial.txt", ".json", ".progress.json"]:
        p = (OUTPUT_DIR / f"{stem}{suffix}").resolve()
        if str(p).startswith(str(OUTPUT_DIR.resolve())) and p.exists():
            p.unlink()
            deleted_output.append(p.name)

    if deleted_output:
        print(f"🗑️  Удалено из output: {', '.join(deleted_output)}")

    return {"ok": True, "deleted_input": filename if input_file.exists() is False else None,
            "deleted_output": deleted_output}
