"""
Шаг 3. Метрики + типология ошибок.

Считаем четыре WER, и разница между ними — главный результат работы:

  1. WER_raw      — без нормализации (как есть)
  2. WER_norm     — после нормализации регистра/пунктуации
                    (raw - norm) = сколько "ошибок" создала метрика, а не модель
  3. WER_fold     — плюс схлопывание ә/ө/ұ/ү/і/ң/қ/ғ в русские аналоги
                    (norm - fold) = цена казахской графики
  4. WER_stem     — плюс срезание аффиксов
                    (norm - stem) = цена агглютинативной морфологии

Плюс CER и разбор замен по типам.

Запуск:
  python src/evaluate.py --hyp results/hyp_large-v3_kk.tsv
"""

import argparse
import csv
import difflib
import json
import os
from collections import Counter

import jiwer

from kk_text import KK_FOLD_MAP, has_kk_letter, normalize, stem, stem_text


def load(data_dir, hyp_path):
    with open(os.path.join(data_dir, "refs.tsv"), encoding="utf-8") as f:
        refs = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    with open(hyp_path, encoding="utf-8") as f:
        hyps = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    ids = [i for i in refs if i in hyps]
    return ids, refs, hyps


def wer_safe(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return float("nan")
    r, h = zip(*pairs)
    return jiwer.wer(list(r), list(h))


def cer_safe(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return float("nan")
    r, h = zip(*pairs)
    return jiwer.cer(list(r), list(h))


def classify(ref_w: str, hyp_w: str) -> str:
    """Классифицируем одну замену слова."""
    if ref_w.isdigit() or hyp_w.isdigit():
        return "числа/нормализация"

    r_fold = "".join(KK_FOLD_MAP.get(c, c) for c in ref_w)
    h_fold = "".join(KK_FOLD_MAP.get(c, c) for c in hyp_w)
    if r_fold == h_fold:
        return "спецбуквы (ә/ө/ұ/ү/і/ң/қ/ғ)"

    if stem(ref_w) == stem(hyp_w) and stem(ref_w) != ref_w:
        return "аффикс (морфология)"

    if r_fold != h_fold and stem(r_fold) == stem(h_fold):
        return "спецбуквы + аффикс"

    ratio = difflib.SequenceMatcher(None, ref_w, hyp_w).ratio()
    if ratio >= 0.6:
        return "близкая форма (фонетика/опечатка)"

    if has_kk_letter(ref_w) and not has_kk_letter(hyp_w):
        return "уход в русскую графику"

    return "полная замена слова"


def char_confusions(ref_w, hyp_w, counter):
    sm = difflib.SequenceMatcher(None, ref_w, hyp_w)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for a, b in zip(ref_w[i1:i2], hyp_w[j1:j2]):
                counter[f"{a} -> {b}"] += 1
        elif tag == "delete":
            for a in ref_w[i1:i2]:
                counter[f"{a} -> ∅"] += 1
        elif tag == "insert":
            for b in hyp_w[j1:j2]:
                counter[f"∅ -> {b}"] += 1


def main(args):
    ids, refs, hyps = load(args.data, args.hyp)

    raw_r = [refs[i]["ref"] for i in ids]
    raw_h = [hyps[i]["hyp"] for i in ids]
    norm_r = [normalize(x) for x in raw_r]
    norm_h = [normalize(x) for x in raw_h]
    fold_r = [normalize(x, fold_kk=True) for x in raw_r]
    fold_h = [normalize(x, fold_kk=True) for x in raw_h]
    stem_r = [stem_text(x) for x in norm_r]
    stem_h = [stem_text(x) for x in norm_h]

    metrics = {
        "n_utt": len(ids),
        "audio_min": round(sum(float(refs[i]["duration_sec"]) for i in ids) / 60, 2),
        "WER_raw": round(wer_safe(raw_r, raw_h), 4),
        "WER_norm": round(wer_safe(norm_r, norm_h), 4),
        "WER_fold": round(wer_safe(fold_r, fold_h), 4),
        "WER_stem": round(wer_safe(stem_r, stem_h), 4),
        "CER_norm": round(cer_safe(norm_r, norm_h), 4),
    }
    metrics["delta_normalization"] = round(metrics["WER_raw"] - metrics["WER_norm"], 4)
    metrics["delta_kk_graphemes"] = round(metrics["WER_norm"] - metrics["WER_fold"], 4)
    metrics["delta_morphology"] = round(metrics["WER_norm"] - metrics["WER_stem"], 4)

    # --- типология ошибок по выравниванию ---
    types = Counter()
    chars = Counter()
    examples = []

    out = jiwer.process_words(norm_r, norm_h)
    for k, chunks in enumerate(out.alignments):
        r_words = norm_r[k].split()
        h_words = norm_h[k].split()
        for ch in chunks:
            if ch.type == "substitute":
                for a, b in zip(
                    r_words[ch.ref_start_idx : ch.ref_end_idx],
                    h_words[ch.hyp_start_idx : ch.hyp_end_idx],
                ):
                    t = classify(a, b)
                    types[t] += 1
                    char_confusions(a, b, chars)
                    examples.append(
                        {"utt": ids[k], "type": t, "ref": a, "hyp": b}
                    )
            elif ch.type == "delete":
                for a in r_words[ch.ref_start_idx : ch.ref_end_idx]:
                    types["пропуск слова"] += 1
                    examples.append({"utt": ids[k], "type": "пропуск слова", "ref": a, "hyp": ""})
            elif ch.type == "insert":
                for b in h_words[ch.hyp_start_idx : ch.hyp_end_idx]:
                    types["вставка слова"] += 1
                    examples.append({"utt": ids[k], "type": "вставка слова", "ref": "", "hyp": b})

    # --- кандидаты в галлюцинации: гипотеза сильно длиннее эталона ---
    halluc = []
    for i in ids:
        r_len = len(normalize(refs[i]["ref"]).split())
        h_len = len(normalize(hyps[i]["hyp"]).split())
        if r_len and h_len > 2 * r_len + 3:
            halluc.append({"id": i, "ref_len": r_len, "hyp_len": h_len,
                           "hyp": hyps[i]["hyp"][:300]})

    # --- смена языка ---
    lang_counter = Counter(
        hyps[i].get("detected_lang", "") for i in ids if hyps[i].get("detected_lang")
    )

    tag = os.path.basename(args.hyp).replace("hyp_", "").replace(".tsv", "")
    os.makedirs(args.results, exist_ok=True)

    with open(os.path.join(args.results, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics,
             "error_types": types.most_common(),
             "char_confusions_top30": chars.most_common(30),
             "detected_lang": lang_counter.most_common(),
             "hallucination_candidates": halluc[:20]},
            f, ensure_ascii=False, indent=2,
        )

    with open(os.path.join(args.results, f"errors_{tag}.tsv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt", "type", "ref", "hyp"], delimiter="\t")
        w.writeheader()
        w.writerows(examples)

    # таблица "что сказали / что распозналось" для отчёта
    with open(os.path.join(args.results, f"table_{tag}.tsv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "reference", "hypothesis", "wer_utt"])
        for i in ids:
            r, h = normalize(refs[i]["ref"]), normalize(hyps[i]["hyp"])
            u = jiwer.wer(r, h) if r.strip() else ""
            w.writerow([i, r, h, round(u, 3) if u != "" else ""])

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nТипы ошибок:")
    total = sum(types.values())
    for t, c in types.most_common():
        print(f"  {c:5d}  {100*c/total:5.1f}%  {t}")
    print("\nТоп подмен символов:")
    for c, n in chars.most_common(15):
        print(f"  {n:5d}  {c}")
    print(f"\nКандидатов в галлюцинации: {len(halluc)}")
    print(f"Определённый язык: {lang_counter.most_common()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hyp", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    main(p.parse_args())
