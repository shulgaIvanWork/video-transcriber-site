"""
transcribe.py — Транскрибация видео/аудио через faster-whisper.

Поддерживаемые форматы: .mp4, .mkv, .mov, .avi, .wav, .mp3, .m4a, .ogg, .flac

Результат:
  output/имя_файла.txt  — чистый текст

Запуск:
  docker compose run --rm transcriber [--model medium] [--lang ru]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from faster_whisper import WhisperModel


def write_txt(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"  ✍️  {path.name}")


def find_media_files(input_dir: Path) -> list[Path]:
    """Ищет видео/аудио файлы в папке (не рекурсивно, топ-уровень)."""
    extensions = {".mp4", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions and p.is_file()
    )
    return files


# ─── Основная логика ───────────────────────────────────────────────


def _write_progress(progress_file: Path, data: dict):
    """Атомарно пишет прогресс в JSON (через temp + rename)."""
    tmp = progress_file.with_suffix(".progress.tmp")
    data["_ts"] = time.time()
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.rename(progress_file)


def _update_partial_txt(txt_path: Path, text_parts: list[str], max_char: int = 50000):
    """Сохраняет частичный текст для превью."""
    partial = "".join(text_parts)
    txt_path.write_text(partial[:max_char], encoding="utf-8")


def transcribe_file(
    model: WhisperModel,
    filepath: Path,
    output_dir: Path,
    language: str = "ru",
    vad: bool = True,
    progress_file: Path | None = None,
):
    """Транскрибирует один файл и сохраняет результат с промежуточным прогрессом."""
    print(f"\n{'='*60}")
    print(f"🎬  {filepath.name}")
    print(f"    размер: {filepath.stat().st_size / 1024 / 1024:.0f} МБ")
    print(f"    язык:   {language}")
    print(f"{'='*60}\n")

    t0 = time.time()
    stem = filepath.stem

    try:
        segments, info = model.transcribe(
            str(filepath),
            language=language,
            beam_size=5,
            vad_filter=vad,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5,
            ),
        )
    except IndexError as e:
        print(f"\n  ❌  В файле нет аудиодорожки!")
        print(f"      faster-whisper не может обработать видео без звука.")
        print(f"      Ошибка: {e}")
        if progress_file:
            _write_progress(progress_file, {"status": "error", "error": f"Нет аудиодорожки: {e}"})
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌  Ошибка при открытии файла: {e}")
        if progress_file:
            _write_progress(progress_file, {"status": "error", "error": str(e)})
        sys.exit(1)

    total_duration = info.duration
    print(f"  📊  Длительность: {total_duration:.0f} с ({total_duration / 3600:.1f} ч)")
    print(f"      Язык:         {info.language} (prob: {info.language_probability:.2f})")

    # Начальный прогресс
    partial_txt_path = output_dir / f"{stem}.partial.txt"
    if progress_file:
        _write_progress(progress_file, {
            "status": "processing",
            "progress_pct": 0,
            "total_duration_sec": round(total_duration, 1),
            "processed_sec": 0,
            "elapsed_sec": 0,
            "segments_count": 0,
            "partial_chars": 0,
            "text": "",
        })
        _update_partial_txt(partial_txt_path, [], max_char=50000)

    # Обрабатываем сегменты в потоке
    text_parts: list[str] = []
    seg_count = 0
    last_progress_save = 0.0
    last_partial_save = 0.0

    try:
        for seg in segments:
            seg_text = seg.text.strip()
            if not seg_text:
                continue

            text_parts.append(seg_text + " ")
            seg_count += 1

            now = time.time()
            processed_sec = seg.end
            progress_pct = min((processed_sec / total_duration) * 100, 99.9)

            # Сохраняем прогресс каждые ~3 сек или 5% прогресса
            if progress_file and (
                now - last_progress_save > 3
                or progress_pct - last_progress_save > 5
            ):
                _write_progress(progress_file, {
                    "status": "processing",
                    "progress_pct": round(progress_pct, 1),
                    "total_duration_sec": round(total_duration, 1),
                    "processed_sec": round(processed_sec, 1),
                    "elapsed_sec": round(now - t0, 1),
                    "segments_count": seg_count,
                    "partial_chars": sum(len(p) for p in text_parts),
                    "text": seg_text,
                })
                last_progress_save = now

            # Сохраняем частичный текст каждые 30 сек
            if progress_file and now - last_partial_save > 30:
                _update_partial_txt(partial_txt_path, text_parts, max_char=50000)
                last_partial_save = now

    except IndexError as e:
        print(f"\n  ❌  В файле нет аудиодорожки!")
        if progress_file:
            _write_progress(progress_file, {"status": "error", "error": f"Нет аудиодорожки: {e}"})
        sys.exit(1)

    t1 = time.time()
    elapsed = t1 - t0
    speed = total_duration / elapsed if elapsed > 0 else 0
    print(f"\n  ⏱️   Обработано за {elapsed / 60:.1f} мин ({speed:.1f}x realtime)")

    # ── Финальное сохранение ──
    full_text = "".join(text_parts).rstrip()

    write_txt(output_dir / f"{stem}.txt", full_text)
    write_txt(partial_txt_path, full_text)

    # Финальный прогресс + чистка временных файлов
    if progress_file:
        _write_progress(progress_file, {
            "status": "done",
            "progress_pct": 100,
            "total_duration_sec": round(total_duration, 1),
            "processed_sec": round(total_duration, 1),
            "elapsed_sec": round(elapsed, 1),
            "segments_count": seg_count,
            "partial_chars": len(full_text),
            "text": "",
        })
        # Финальный partial.txt = полный текст (уже сохранён выше как .txt)
        # Удаляем временные файлы прогресса
        try:
            progress_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            partial_txt_path.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"\n  ✅  {stem} — готово!\n")


# ─── CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Транскрибация видео/аудио через faster-whisper"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="/app/input",
        help="Файл или папка с видео (по умолч. /app/input)",
    )
    parser.add_argument(
        "-o", "--output",
        default="/app/output",
        help="Папка для результатов (по умолч. /app/output)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL", "base"),
        help="Размер модели: tiny, base, small, medium, large-v3 (по умолч. base)",
    )
    parser.add_argument(
        "--lang",
        default=os.environ.get("WHISPER_LANG", "ru"),
        help="Язык (по умолч. ru)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("WHISPER_DEVICE", "cpu"),
        choices=["cpu", "cuda"],
        help="Устройство: cpu или cuda (по умолч. cpu)",
    )
    parser.add_argument(
        "--compute",
        default=os.environ.get("WHISPER_COMPUTE", "int8"),
        choices=["float16", "int8_float16", "int8", "float32"],
        help="Тип вычислений (для CPU — int8, для CUDA — float16, по умолч. int8)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_false",
        dest="vad",
        help="Отключить VAD-фильтр (может снизить качество на длинных файлах)",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Путь к JSON-файлу для записи прогресса в реальном времени",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌  Путь не найден: {input_path}")
        sys.exit(1)

    # Определяем файлы для обработки
    if input_path.is_file():
        files = [input_path]
    else:
        files = find_media_files(input_path)
        if not files:
            print(f"⚠️  В папке {input_path} не найдено видео/аудио файлов.")
            print(f"    Поддерживаемые форматы: .mp4 .mkv .avi .mov .wav .mp3 .m4a .ogg .flac")
            sys.exit(1)

    print(f"🚀  faster-whisper ({args.model}) — {len(files)} файл(ов)")
    print(f"    device:  {args.device}")
    print(f"    compute: {args.compute}")
    print(f"    язык:    {args.lang}")
    print(f"    VAD:     {'вкл' if args.vad else 'выкл'}")
    print()

    # Загружаем модель
    print(f"⏳  Загружаю модель '{args.model}'...")
    t_load = time.time()
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute,
        cpu_threads=os.cpu_count() or 4,
        num_workers=2,
    )
    print(f"   ✅  Загрузка за {time.time() - t_load:.0f} с\n")

    # Обрабатываем
    output_path.mkdir(parents=True, exist_ok=True)

    for filepath in files:
        progress_file = Path(args.progress_file) if args.progress_file else None
        transcribe_file(model, filepath, output_path, args.lang, args.vad, progress_file=progress_file)

    print(f"📁  Все результаты: {output_path.resolve()}")


if __name__ == "__main__":
    main()
