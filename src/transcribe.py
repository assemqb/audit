"""
Step 2. Run ASR over the prepared wav files.

Uses faster-whisper. We log not just the text but also no_speech_prob,
avg_logprob, and timing. That's useful later for spotting hallucinations
and for the "how much can we trust these numbers" section.

Example:
  python src/transcribe.py --model large-v3 --lang kk
  python src/transcribe.py --model small   --lang kk
  python src/transcribe.py --model large-v3 --lang None   # auto-detect language
"""

import argparse
import csv
import os
import time

import ctranslate2
from faster_whisper import WhisperModel


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def main(args):
    refs_path = os.path.join(args.data, "refs.tsv")
    with open(refs_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    device = resolve_device(args.device)
    compute = args.compute_type or ("int8" if device == "cpu" else "int8_float16")

    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute,
    )

    lang = None if args.lang.lower() == "none" else args.lang

    out_rows = []
    total_audio = 0.0
    t_start = time.time()

    for i, row in enumerate(rows, 1):
        wav = os.path.join(args.data, "audio", f"{row['id']}.wav")
        segments, info = model.transcribe(
            wav,
            language=lang,
            beam_size=args.beam_size,
            vad_filter=False,          # deliberately no VAD: we want to see hallucinations
            condition_on_previous_text=False,
        )
        segments = list(segments)
        hyp = " ".join(s.text.strip() for s in segments).strip()

        avg_logprob = (
            sum(s.avg_logprob for s in segments) / len(segments) if segments else None
        )
        no_speech = (
            max(s.no_speech_prob for s in segments) if segments else None
        )

        out_rows.append(
            {
                "id": row["id"],
                "hyp": hyp,
                "detected_lang": info.language,
                "lang_prob": round(info.language_probability, 4),
                "avg_logprob": round(avg_logprob, 4) if avg_logprob is not None else "",
                "max_no_speech_prob": round(no_speech, 4) if no_speech is not None else "",
                "n_segments": len(segments),
            }
        )
        total_audio += float(row["duration_sec"])
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}")

    elapsed = time.time() - t_start

    tag = f"{args.model}_{args.lang}".replace("/", "-")
    out_path = os.path.join(args.results, f"hyp_{tag}.tsv")
    os.makedirs(args.results, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nDone: {out_path}")
    print(f"Audio: {total_audio/60:.2f} min, elapsed: {elapsed/60:.2f} min")
    print(f"RTF: {elapsed/total_audio:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="large-v3")
    p.add_argument("--lang", default="kk")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--compute_type", default=None)
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    main(p.parse_args())
