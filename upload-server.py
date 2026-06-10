#!/usr/bin/env python3
"""
upload-server.py — FastAPI сервер транскрибации видео → текст.

Запуск:
    uvicorn upload-server:app --host 0.0.0.0 --port 8080 --workers 1

Или внутри Docker:
    docker run -p 8080:8080 video-transcriber
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# ─── Конфиг ────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
PORT = int(os.environ.get("PORT", 8080))
HOST = os.environ.get("HOST", "0.0.0.0")
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 ГБ
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="🎬 Video → Text Transcriber",
    version="3.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Очередь ────────────────────────────────────────────────────────

class TranscriptionQueue:
    """Простая очередь: один файл обрабатывается, остальные ждут."""

    def __init__(self, max_concurrent: int = 1):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: list[str] = []
        self._jobs: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, filename: str) -> int:
        async with self._lock:
            self._queue.append(filename)
            position = len(self._queue)
            self._jobs[filename] = {
                "status": "queued",
                "position": position,
                "started": None,
            }
            return position

    async def start_job(self, filename: str):
        async with self._lock:
            self._jobs[filename] = {
                "status": "processing",
                "started": time.time(),
                "progress_file": str(OUTPUT_DIR / f"{Path(filename).stem}.progress.json"),
            }

    async def finish_job(self, filename: str, success: bool, error_msg: str = ""):
        async with self._lock:
            if success:
                self._jobs[filename] = {"status": "done"}
            else:
                self._jobs[filename] = {"status": "error", "error": error_msg}
            if filename in self._queue:
                self._queue.remove(filename)

    async def get_job(self, filename: str) -> Optional[dict]:
        async with self._lock:
            job = self._jobs.get(filename)
            if job is None:
                return None
            return dict(job)

    async def queue_size(self) -> int:
        async with self._lock:
            return len(self._queue)


queue = TranscriptionQueue(max_concurrent=MAX_CONCURRENT)


# ─── Фоновый воркер ────────────────────────────────────────────────

async def process_file(filename: str):
    """Запускает транскрибацию в подпроцессе и следит за результатом."""
    stem = Path(filename).stem
    progress_file = OUTPUT_DIR / f"{stem}.progress.json"
    input_path = INPUT_DIR / filename
    output_files = [OUTPUT_DIR / f"{stem}.txt"]

    await queue.start_job(filename)
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
            # Проверяем, что хотя бы один выходной файл создан
            any_output = any(p.exists() for p in output_files)
            if any_output:
                await queue.finish_job(filename, success=True)
                print(f"✅  Готово: {filename}")

                # Чистим input после успешной обработки
                try:
                    input_path.unlink(missing_ok=True)
                    print(f"🧹  Input удалён: {filename}")
                except Exception as e:
                    print(f"⚠️  Не удалось удалить {filename}: {e}")

                # Чистим progress-файл (если остался)
                try:
                    progress_file.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                err = "Файлы результата не найдены"
                await queue.finish_job(filename, success=False, error_msg=err)
                print(f"❌  {err}: {filename}")
        else:
            err_text = stderr.decode("utf-8", errors="replace")[-500:] if stderr else ""
            if not err_text:
                err_text = stdout.decode("utf-8", errors="replace")[-500:] if stdout else "Ошибка транскрибации"
            await queue.finish_job(filename, success=False, error_msg=err_text)
            print(f"❌  Ошибка ({proc.returncode}): {err_text[:200]}")

    except asyncio.TimeoutError:
        await queue.finish_job(filename, success=False, error_msg="Таймаут (>24ч)")
        print(f"❌  Таймаут: {filename}")
    except Exception as e:
        await queue.finish_job(filename, success=False, error_msg=str(e))
        print(f"❌  {e}")


async def worker_loop():
    """Фоновый цикл: берёт задачи из очереди и обрабатывает."""
    while True:
        filename = None
        async with queue._lock:
            if queue._queue:
                # Берём первый файл, который ещё не в processing
                for f in queue._queue:
                    job = queue._jobs.get(f, {})
                    if job.get("status") == "queued":
                        filename = f
                        break

        if filename:
            await process_file(filename)
        else:
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup():
    # Стартуем фоновый воркер
    asyncio.create_task(worker_loop())

    # Подхватываем файлы, которые могли остаться в input после перезапуска
    existing = [f.name for f in sorted(INPUT_DIR.iterdir()) if f.is_file()]
    new_count = 0
    for fname in existing:
        stem = Path(fname).stem
        # Если нет результата — ставим в очередь
        if not (OUTPUT_DIR / f"{stem}.txt").exists():
            await queue.enqueue(fname)
            print(f"↩  Подхвачен из input: {fname}")
            new_count += 1

    print(f"🎬  Сервер запущен: http://localhost:{PORT}")
    if new_count:
        print(f"📦  Добавлено в очередь: {new_count} файл(ов) из input/")


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
async def health():
    return {"status": "ok", "queue": await queue.queue_size()}


@app.get("/files")
async def list_files():
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
      const resp = await fetch('/status?file=' + encodeURIComponent(filename));
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

  fetch('/preview?file=' + encodeURIComponent(filename))
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
  actions.innerHTML = `
    <div class="flex">
      <a class="btn btn-primary" href="/download/${stem}.txt" download>📄 Скачать текст (.txt)</a>
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
  fetch('/files')
    .then(r => r.json())
    .then(data => {
      const body = document.getElementById('filesBody');
      const count = document.getElementById('filesCount');
      const files = data.files || [];

      count.textContent = files.length;

      if (!files.length) {
        body.innerHTML = '<div class="files-empty">Пока нет готовых расшифровок</div>';
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
        return `<div class="file-row">
          <span class="fname">📄 ${escHtml(f.name)}</span>
          <span class="finfo">${sizeStr} · ${dateStr}</span>
          <a class="fdl" href="/download/${encodeURIComponent(f.name)}" download>Скачать</a>
        </div>`;
      }).join('');
    })
    .catch(() => {});
}

// Загружаем список при старте и после завершения задачи
loadFilesList();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


# ─── Upload ─────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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
            chunk = await file.read(64 * 1024)  # 64KB чанки
            if not chunk:
                break
            f.write(chunk)

    # Ставим в очередь
    position = await queue.enqueue(dest.name)

    print(f"📥  Загружен: {dest.name} ({dest.stat().st_size / 1024 / 1024:.0f} МБ), "
          f"позиция в очереди: {position}")

    return {
        "ok": True,
        "name": dest.name,
        "size_mb": round(dest.stat().st_size / (1024 * 1024), 1),
        "position": position,
    }


# ─── Status ─────────────────────────────────────────────────────────

@app.get("/status")
async def get_status(file: str):
    """Статус обработки файла."""
    stem = get_stem(file)

    # Сначала проверяем очередь
    job = await queue.get_job(file)
    if job is None:
        # Возможно файл уже готов (принесли в input вручную)
        txt = OUTPUT_DIR / f"{stem}.txt"
        if txt.exists():
            return {"status": "done", "txt": f"{stem}.txt"}
        return {"status": "idle"}

    if job["status"] == "queued":
        return {
            "status": "queued",
            "position": job.get("position", 0),
        }

    if job["status"] == "processing":
        elapsed = time.time() - (job.get("started") or time.time())

        # Читаем прогресс из файла
        pf = job.get("progress_file")
        progress_data = {}
        if pf and os.path.isfile(pf):
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)
            except Exception:
                pass

        pct = progress_data.get("progress_pct", 0)
        processed_sec = progress_data.get("processed_sec", 0)
        total_sec = progress_data.get("total_duration_sec", 0)
        seg_count = progress_data.get("segments_count", 0)
        last_text = progress_data.get("text", "")

        return {
            "status": "processing",
            "progress_pct": round(pct, 1),
            "processed_sec": processed_sec,
            "total_duration_sec": total_sec,
            "segments_count": seg_count,
            "last_text": last_text[:200] if last_text else "",
            "elapsed_sec": round(elapsed, 1),
            "has_partial": os.path.isfile(OUTPUT_DIR / f"{stem}.partial.txt"),
        }

    if job["status"] == "done":
        size_kb = 0
        txt_p = OUTPUT_DIR / f"{stem}.txt"
        if txt_p.exists():
            size_kb = round(txt_p.stat().st_size / 1024, 1)
        return {"status": "done", "txt": f"{stem}.txt", "size_kb": size_kb}

    return {
        "status": "error",
        "error": job.get("error", "Неизвестная ошибка"),
    }


# ─── Preview ───────────────────────────────────────────────────────

@app.get("/preview")
async def get_preview(file: str):
    """Частичный текст в процессе обработки."""
    stem = get_stem(file)
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
async def download_file(filename: str):
    """Скачать готовый файл."""
    # security: не даём выйти за output/
    filepath = (OUTPUT_DIR / filename).resolve()
    if not str(filepath).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(status_code=403, detail="forbidden")

    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    mime_map = {
        ".txt": "text/plain; charset=utf-8",
    }
    content_type = mime_map.get(filepath.suffix, "application/octet-stream")

    data = filepath.read_bytes()
    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filepath.name}"',
            "Content-Length": str(len(data)),
        },
    )
