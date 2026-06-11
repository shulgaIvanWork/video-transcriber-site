"""
transcribe.py — Транскрибация видео/аудио через faster-whisper.

Поддерживаемые форматы: .mp4, .mkv, .mov, .avi, .wav, .mp3, .m4a, .ogg, .flac

Для длинных файлов (> 30 мин) — нарезка на 10-минутные чанки,
транскрибация каждого по отдельности, склейка результата.

Результат:
  output/имя_файла.txt  — чистый текст

Запуск:
  docker compose run --rm transcriber [--model medium] [--lang ru]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from faster_whisper import WhisperModel

# ─── Конфиг ────────────────────────────────────────────────────────

CHUNK_SECONDS = 600  # 10 минут на чанк (макс аудио в память за раз)
LONG_FILE_THRESHOLD = 1800  # 30 минут — порог, после которого режем на чанки


def write_txt(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"  ✍️  {path.name}")


def find_media_files(input_dir: Path) -> list[Path]:
    extensions = {".mp4", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    return sorted(p for p in input_dir.iterdir()
                  if p.suffix.lower() in extensions and p.is_file())


# ─── Прогресс ──────────────────────────────────────────────────────


def _write_progress(progress_file: Path, data: dict):
    tmp = progress_file.with_name(progress_file.name + ".tmp")
    data["_ts"] = time.time()
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.rename(progress_file)


def _update_partial_txt(txt_path: Path, text_parts: list[str], max_char: int = 50000):
    partial = "".join(text_parts)
    txt_path.write_text(partial[:max_char], encoding="utf-8")


# ─── Вспомогательные ffmpeg ────────────────────────────────────────


def ffmpeg_run(cmd: list[str], desc: str = ""):
    """Запускает ffmpeg, ловит ошибки."""
    print(f"  🎞️  {desc or ' '.join(cmd[:3])}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ❌ ffmpeg ошибка: {r.stderr[-300:]}")
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-200]}")


def get_audio_duration(audio_path: Path) -> float:
    """Длительность аудио в секундах."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def extract_audio_wav(filepath: Path, dst: Path):
    """Извлекает аудиодорожку как 16 кГц моно WAV."""
    ffmpeg_run([
        "ffmpeg", "-y", "-i", str(filepath),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(dst),
    ], desc=f"Извлечение аудио: {filepath.name}")


# ─── Потоковая транскрибация (весь файл в памяти) ───────────────────


def transcribe_stream(
    model: WhisperModel, filepath: Path, output_dir: Path,
    language: str, vad: bool, progress_file: Path | None,
    t0: float, text_parts: list, partial_txt_path: Path,
):
    """Транскрибация всего файла целиком (в потоке сегментов)."""
    stem = filepath.stem

    segments, info = model.transcribe(
        str(filepath),
        language=language,
        beam_size=5,
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
    )

    total_duration = info.duration
    print(f"  📊  Длительность: {total_duration:.0f} с ({total_duration / 3600:.1f} ч)")
    print(f"      Язык:         {info.language} (prob: {info.language_probability:.2f})")

    seg_count = 0
    last_progress_save = 0.0
    last_partial_save = 0.0

    if progress_file:
        _write_progress(progress_file, {
            "status": "processing",
            "progress_pct": 0,
            "total_duration_sec": round(total_duration, 1),
            "processed_sec": 0,
            "elapsed_sec": 0, "segments_count": 0, "text": "",
        })

    for seg in segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue
        text_parts.append(seg_text + " ")
        seg_count += 1

        now = time.time()
        processed_sec = seg.end
        progress_pct = min((processed_sec / total_duration) * 100, 99.9)

        if progress_file and (
            now - last_progress_save > 3 or progress_pct - last_progress_save > 5
        ):
            _write_progress(progress_file, {
                "status": "processing",
                "progress_pct": round(progress_pct, 1),
                "total_duration_sec": round(total_duration, 1),
                "processed_sec": round(processed_sec, 1),
                "elapsed_sec": round(now - t0, 1),
                "segments_count": seg_count,
                "text": seg_text,
            })
            last_progress_save = now

        if progress_file and now - last_partial_save > 30:
            _update_partial_txt(partial_txt_path, text_parts, max_char=50000)
            last_partial_save = now

    return seg_count


# ─── Чанковая транскрибация ─────────────────────────────────────────


def transcribe_chunked(
    model: WhisperModel, filepath: Path, output_dir: Path,
    language: str, vad: bool, progress_file: Path | None,
    t0: float, text_parts: list, partial_txt_path: Path,
):
    """
    Для больших файлов:
    1. Извлекаем аудио → 16 кГц моно WAV
    2. Режем на WAV-чанки по CHUNK_SECONDS
    3. Каждый чанк транскрибируем отдельно
    4. Склеиваем результаты
    """
    stem = filepath.stem
    tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_chunks_"))

    try:
        # Шаг 1: извлекаем аудио
        audio_wav = tmp_dir / "audio.wav"
        extract_audio_wav(filepath, audio_wav)

        total_duration = get_audio_duration(audio_wav)
        total_chunks = max(1, int(total_duration // CHUNK_SECONDS) +
                           (1 if total_duration % CHUNK_SECONDS > 0 else 0))
        print(f"  📊  Длительность: {total_duration:.0f} с ({total_duration/3600:.1f} ч)")
        print(f"  🧩  Чанков: {total_chunks} по {CHUNK_SECONDS} с")

        # Шаг 2: режем на чанки
        chunk_pattern = str(tmp_dir / "chunk_%04d.wav")
        ffmpeg_run([
            "ffmpeg", "-y", "-i", str(audio_wav),
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            chunk_pattern,
        ], desc="Нарезка на чанки")

        chunk_files = sorted(tmp_dir.glob("chunk_*.wav"))
        if not chunk_files:
            # Если не нарезалось — обрабатываем как есть
            chunk_files = [audio_wav]
            total_chunks = 1

        print(f"  🧩  Фактических чанков: {len(chunk_files)}")

        all_seg_count = 0
        processed_chunk_sec = 0

        for ci, chunk_path in enumerate(chunk_files):
            chunk_dur = get_audio_duration(chunk_path)
            chunk_label = f"[{ci+1}/{len(chunk_files)}]"
            print(f"\n  {chunk_label} 🎯 {chunk_path.name} ({chunk_dur:.0f} с)")

            segments, info = model.transcribe(
                str(chunk_path),
                language=language,
                beam_size=5,
                vad_filter=vad,
                vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
            )

            chunk_text_parts = []
            seg_count = 0
            last_save = 0.0

            for seg in segments:
                seg_text = seg.text.strip()
                if not seg_text:
                    continue
                chunk_text_parts.append(seg_text + " ")
                text_parts.append(seg_text + " ")
                seg_count += 1
                all_seg_count += 1

                now = time.time()
                # Общий прогресс = (сделанные чанки + прогресс внутри чанка) / всего
                chunk_progress = seg.end / chunk_dur if chunk_dur > 0 else 1
                overall_pct = ((ci + chunk_progress) / total_chunks) * 100
                processed_total_sec = processed_chunk_sec + seg.end

                if progress_file and (now - last_save > 3 or seg.end % 30 < 1):
                    _write_progress(progress_file, {
                        "status": "processing",
                        "progress_pct": round(overall_pct, 1),
                        "total_duration_sec": round(total_duration, 1),
                        "processed_sec": round(processed_total_sec, 1),
                        "elapsed_sec": round(now - t0, 1),
                        "segments_count": all_seg_count,
                        "chunk": f"{ci+1}/{len(chunk_files)}",
                        "text": seg_text,
                    })
                    last_save = now

                if progress_file and now - last_save > 30:
                    _update_partial_txt(partial_txt_path, text_parts, max_char=50000)

            # Частичный текст после каждого чанка
            _update_partial_txt(partial_txt_path, text_parts, max_char=50000)
            processed_chunk_sec += chunk_dur

            # Прогресс после чанка
            if progress_file:
                _write_progress(progress_file, {
                    "status": "processing",
                    "progress_pct": round(((ci + 1) / total_chunks) * 100, 1),
                    "total_duration_sec": round(total_duration, 1),
                    "processed_sec": round(processed_chunk_sec, 1),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "segments_count": all_seg_count,
                    "chunk": f"{ci+1}/{len(chunk_files)}",
                    "text": chunk_text_parts[-1][:200] if chunk_text_parts else "",
                })

    finally:
        # Чистим временную директорию
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ─── Основная логика ────────────────────────────────────────────────


def transcribe_file(
    model: WhisperModel,
    filepath: Path,
    output_dir: Path,
    language: str = "ru",
    vad: bool = True,
    progress_file: Path | None = None,
):
    """Транскрибирует один файл. Если длинный — с нарезкой на чанки."""
    print(f"\n{'=' * 60}")
    print(f"🎬  {filepath.name}")
    print(f"    размер: {filepath.stat().st_size / 1024 / 1024:.0f} МБ")
    print(f"    язык:   {language}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    stem = filepath.stem
    partial_txt_path = output_dir / f"{stem}.partial.txt"
    text_parts: list[str] = []

    try:
        # Быстрая проверка длительности через ffprobe
        do_chunk = False
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
                capture_output=True, text=True, timeout=30,
            )
            if probe.returncode == 0:
                dur = float(probe.stdout.strip())
                if dur > LONG_FILE_THRESHOLD:
                    do_chunk = True
                    print(f"  ⏱️   Длительность: {dur:.0f} с ({dur/3600:.1f} ч) — включаю чанковый режим")
        except Exception:
            pass  # если ffprobe не сработал — транскрибируем как есть

        if do_chunk:
            transcribe_chunked(
                model, filepath, output_dir, language, vad,
                progress_file, t0, text_parts, partial_txt_path,
            )
        else:
            transcribe_stream(
                model, filepath, output_dir, language, vad,
                progress_file, t0, text_parts, partial_txt_path,
            )

    except IndexError as e:
        print(f"\n  ❌  В файле нет аудиодорожки!")
        print(f"      Ошибка: {e}")
        if progress_file:
            _write_progress(progress_file, {"status": "error",
                            "error": f"Нет аудиодорожки: {e}"})
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌  Ошибка: {e}")
        if progress_file:
            _write_progress(progress_file, {"status": "error", "error": str(e)})
        sys.exit(1)

    t1 = time.time()
    elapsed = t1 - t0

    # ── Финальное сохранение ──
    full_text = "".join(text_parts).rstrip()
    write_txt(output_dir / f"{stem}.txt", full_text)
    write_txt(partial_txt_path, full_text)

    if progress_file:
        total_dur = 0
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
                capture_output=True, text=True, timeout=10,
            )
            if probe.returncode == 0:
                total_dur = float(probe.stdout.strip())
        except Exception:
            pass

        _write_progress(progress_file, {
            "status": "done", "progress_pct": 100,
            "total_duration_sec": round(total_dur, 1),
            "processed_sec": round(total_dur, 1),
            "elapsed_sec": round(elapsed, 1),
            "segments_count": 0, "text": "",
        })
        try:
            progress_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            partial_txt_path.unlink(missing_ok=True)
        except Exception:
            pass

    speed = total_dur / elapsed if elapsed > 0 else 0
    print(f"\n  ⏱️   Обработано за {elapsed/60:.1f} мин ({speed:.1f}x realtime)")
    print(f"\n  ✅  {stem} — готово!\n")


# ─── CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Транскрибация видео/аудио через faster-whisper"
    )
    parser.add_argument(
        "input", nargs="?", default="/app/input",
        help="Файл или папка с видео (по умолч. /app/input)",
    )
    parser.add_argument(
        "-o", "--output", default="/app/output",
        help="Папка для результатов (по умолч. /app/output)",
    )
    parser.add_argument(
        "--model", default=os.environ.get("WHISPER_MODEL", "base"),
        help="Размер модели: tiny, base, small, medium, large-v3 (по умолч. base)",
    )
    parser.add_argument(
        "--lang", default=os.environ.get("WHISPER_LANG", "ru"),
        help="Язык (по умолч. ru)",
    )
    parser.add_argument(
        "--device", default=os.environ.get("WHISPER_DEVICE", "cpu"),
        choices=["cpu", "cuda"], help="Устройство: cpu или cuda (по умолч. cpu)",
    )
    parser.add_argument(
        "--compute", default=os.environ.get("WHISPER_COMPUTE", "int8"),
        choices=["float16", "int8_float16", "int8", "float32"],
        help="Тип вычислений (для CPU — int8, для CUDA — float16, по умолч. int8)",
    )
    parser.add_argument(
        "--no-vad", action="store_false", dest="vad",
        help="Отключить VAD-фильтр",
    )
    parser.add_argument(
        "--progress-file", default=None,
        help="Путь к JSON-файлу для записи прогресса",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌  Путь не найден: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        files = [input_path]
    else:
        files = find_media_files(input_path)
        if not files:
            print(f"⚠️  В папке {input_path} не найдено видео/аудио файлов.")
            sys.exit(1)

    print(f"🚀  faster-whisper ({args.model}) — {len(files)} файл(ов)")
    print(f"    device:  {args.device}")
    print(f"    compute: {args.compute}")
    print(f"    язык:    {args.lang}")
    print(f"    VAD:     {'вкл' if args.vad else 'выкл'}")
    print()

    print(f"⏳  Загружаю модель '{args.model}'...")
    t_load = time.time()
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute,
        cpu_threads=2,
        num_workers=1,
    )
    print(f"   ✅  Загрузка за {time.time() - t_load:.0f} с\n")

    output_path.mkdir(parents=True, exist_ok=True)
    for filepath in files:
        progress_file = Path(args.progress_file) if args.progress_file else None
        transcribe_file(model, filepath, output_path, args.lang, args.vad,
                        progress_file=progress_file)

    print(f"📁  Все результаты: {output_path.resolve()}")


if __name__ == "__main__":
    main()
