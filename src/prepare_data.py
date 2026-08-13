"""
Step 1. Build a test set of Kazakh speech.

Two sources to pick from (--source), both technically "read" speech. No
honest streamable "spontaneous" Kazakh corpus was found on HF
(issai/Kazakh_Speech_Corpus_2 has TV/radio/podcast audio, which is closer to
spontaneous speech, but ships as one unprocessed 80+ GB tar.gz with no split
into utterances, not practical for a quick audit):

  fleurs        google/fleurs kk_kz, studio narrator reading, clean
                recording conditions, has FLEURS' own normalized transcription
  common_voice  Common Voice kk, crowdsourced: different mics, devices,
                accents, but the text is still read from a prompt, not spontaneous

The difference between them isn't "speech vs spontaneous speech", it's
"clean studio vs noisy everyday recording conditions". That's still a useful
angle: it shows how much the model relies on studio-quality input.

Downloaded in streaming mode so we don't pull the whole corpus.

Output:
  <out>/audio/*.wav   audio as provided by the source (Whisper resamples it itself)
  <out>/refs.tsv      id \t reference \t alt reference \t duration_sec \t source
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

TARGET_MINUTES = 12.0  # a bit more than the required 10 minutes, as a buffer

SOURCES = {
    "fleurs": {
        "dataset": "google/fleurs",
        "config": "kk_kz",
        "text_field": "raw_transcription",
        "text_alt_field": "transcription",  # FLEURS' own normalized transcription
    },
    "common_voice": {
        "dataset": "Shirali/common_voice_11_0_kk",
        "config": None,
        "text_field": "sentence",
        "text_alt_field": None,
    },
}


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


def main(out_dir: str, target_minutes: float, split: str, source: str, limit=None):
    configure_cache(ROOT)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    cfg = SOURCES[source]
    stream_split = split.split("[")[0] if "[" in split else split
    load_args = (cfg["dataset"], cfg["config"]) if cfg["config"] else (cfg["dataset"],)
    ds = load_dataset(*load_args, split=stream_split, streaming=True)

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

        # drop very short scraps, they make the metric noisy
        if dur < 1.0:
            continue

        uid = f"{source}_{i:05d}"
        path = os.path.join(audio_dir, f"{uid}.wav")
        sf.write(path, wav, sr)

        # The main ref keeps case and punctuation exactly as the source has
        # it. Part of ASR "error" is really a normalization mismatch, and we
        # want to show that separately. ref_alt is the source's own
        # alternate normalization, when it has one (FLEURS only).
        rows.append(
            {
                "id": uid,
                "ref": item[cfg["text_field"]],
                "ref_alt": item[cfg["text_alt_field"]] if cfg["text_alt_field"] else "",
                "duration_sec": round(dur, 3),
                "source": source,
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
            fieldnames=["id", "ref", "ref_alt", "duration_sec", "source"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Source: {source} ({cfg['dataset']})")
    print(f"Utterances: {len(rows)}")
    print(f"Total duration: {total_sec / 60:.2f} min")
    print(f"References: {tsv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--minutes", type=float, default=TARGET_MINUTES)
    p.add_argument("--split", default="test")
    p.add_argument("--source", default="fleurs", choices=sorted(SOURCES))
    p.add_argument("--limit", type=int, default=None, help="Optional cap for small smoke tests, e.g. --limit 5")
    args = p.parse_args()
    main(args.out, args.minutes, args.split, args.source, limit=args.limit)
