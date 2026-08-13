"""
Шаг 1. Готовим тестовый набор казахской речи.

Источник: google/fleurs, конфиг kk_kz (читаная студийная речь, есть эталонные
транскрипции). Качаем в режиме streaming, чтобы не тянуть весь корпус.

Результат:
  data/audio/*.wav        — аудио 16 kHz mono
  data/refs.tsv           — id \t эталонная транскрипция \t длительность_сек
"""

import argparse
import csv
import os

# Force repo-local Hugging Face cache before importing datasets / huggingface_hub.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HF_CACHE = os.path.join(ROOT, ".hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "transformers")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import soundfile as sf
from datasets import load_dataset

TARGET_MINUTES = 12.0  # берём с запасом к требуемым 10 минутам


def configure_cache(base_dir: str):
    """Use a repo-local cache to avoid sandbox/permission issues in IDE environments."""
    root = os.path.abspath(base_dir)
    cache_dir = os.path.join(root, ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
    return cache_dir


def main(out_dir: str, target_minutes: float, split: str, limit=None):
    configure_cache(ROOT)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    stream_split = split.split("[")[0] if "[" in split else split
    ds = load_dataset("google/fleurs", "kk_kz", split=stream_split, streaming=True)

    total_sec = 0.0
    rows = []
    seen = 0

    for i, item in enumerate(ds):
        if limit is not None and seen >= limit:
            break

        audio = item["audio"]
        wav = audio["array"]
        sr = audio["sampling_rate"]
        dur = len(wav) / sr

        # отсекаем совсем короткие огрызки — на них метрика шумит
        if dur < 1.0:
            continue

        uid = f"fleurs_{i:05d}"
        path = os.path.join(audio_dir, f"{uid}.wav")
        sf.write(path, wav, sr)

        # raw_transcription сохраняет регистр и пунктуацию — это важно:
        # часть "ошибок" ASR на самом деле разница нормализации, и мы хотим
        # уметь это показать отдельно.
        rows.append(
            {
                "id": uid,
                "ref": item["raw_transcription"],
                "ref_norm_fleurs": item["transcription"],
                "duration_sec": round(dur, 3),
                "source": "fleurs_kk_test",
            }
        )

        total_sec += dur
        seen += 1
        if total_sec >= target_minutes * 60:
            break

        if limit is not None and seen >= limit:
            break

    tsv_path = os.path.join(out_dir, "refs.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "ref", "ref_norm_fleurs", "duration_sec", "source"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Записей: {len(rows)}")
    print(f"Суммарная длительность: {total_sec / 60:.2f} мин")
    print(f"Эталоны: {tsv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--minutes", type=float, default=TARGET_MINUTES)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None, help="Optional cap for small smoke tests, e.g. --limit 5")
    args = p.parse_args()
    main(args.out, args.minutes, args.split, limit=args.limit)
