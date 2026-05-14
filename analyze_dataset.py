#!/usr/bin/env python3
"""
analyze_dataset.py -- статистический анализ диалогов train/test.
Запуск: python analyze_dataset.py --data_dir /path/to/data
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".", help="Папка с train/test json и csv")
    parser.add_argument("--output", default="dataset_analysis.csv")
    return parser.parse_args()


def dialogs_to_flat(dialogs):
    rows = []
    for did, msgs in dialogs.items():
        for m in msgs:
            text = m["text"]
            tokens = text.split()
            cyr = sum(1 for c in text if chr(0x0400) <= c <= chr(0x04FF))
            rows.append({
                "dialog_id":         did,
                "participant_index": m["participant_index"],
                "message_idx":       m["message"],
                "char_len":          len(text),
                "word_count":        len(tokens),
                "cyrillic_ratio":    cyr / max(len(text), 1),
                "upper_ratio":       sum(1 for c in text if c.isupper()) / max(len(text), 1),
                "has_newline":       int("\n" in text),
                "ends_with_q":       int(text.strip().endswith("?")),
                "has_url":           int("http" in text.lower()),
                "text":              text,
            })
    return pd.DataFrame(rows)


def ngram_diversity(texts, n=2):
    all_ngrams = []
    for t in texts:
        toks = t.lower().split()
        all_ngrams += [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return len(set(all_ngrams)) / max(len(all_ngrams), 1)


def vocab_richness(texts):
    tokens = [w for t in texts for w in t.lower().split()]
    return len(set(tokens)) / max(len(tokens), 1)


def self_repeat(texts):
    if len(texts) < 2:
        return 0.0
    return sum(1 for i in range(1, len(texts)) if texts[i] == texts[i-1]) / (len(texts)-1)


def echo_ratio(own_texts, opp_texts):
    if not opp_texts or not own_texts:
        return 0.0
    echoes = 0
    for i, t in enumerate(own_texts):
        prev_opp = opp_texts[i-1] if 0 < i <= len(opp_texts) else (opp_texts[0] if opp_texts else None)
        if prev_opp and (t == prev_opp or t in prev_opp or prev_opp in t):
            echoes += 1
    return echoes / len(own_texts)


def participant_stats(flat_dialog, pidx):
    own = flat_dialog[flat_dialog["participant_index"] == pidx].sort_values("message_idx")
    opp = flat_dialog[flat_dialog["participant_index"] == (1-pidx)]
    ot = own["text"].tolist()
    op = opp["text"].tolist()
    ol = own["char_len"].values

    return {
        "n_utterances":        len(own),
        "n_opp_utterances":    len(opp),
        "total_chars":         int(own["char_len"].sum()),
        "total_words":         int(own["word_count"].sum()),
        "char_len_mean":       float(np.mean(ol)) if len(ol) else 0.0,
        "char_len_std":        float(np.std(ol))  if len(ol) > 1 else 0.0,
        "char_len_min":        float(np.min(ol))  if len(ol) else 0.0,
        "char_len_max":        float(np.max(ol))  if len(ol) else 0.0,
        "char_len_median":     float(np.median(ol)) if len(ol) else 0.0,
        "char_len_cv":         float(np.std(ol)/(np.mean(ol)+1e-6)) if len(ol) > 1 else 0.0,
        "word_count_mean":     float(own["word_count"].mean()) if len(own) else 0.0,
        "word_count_std":      float(own["word_count"].std())  if len(own) > 1 else 0.0,
        "vocab_richness":      vocab_richness(ot),
        "bigram_diversity":    ngram_diversity(ot, 2),
        "trigram_diversity":   ngram_diversity(ot, 3),
        "self_repeat_ratio":   self_repeat(ot),
        "echo_ratio":          echo_ratio(ot, op),
        "cyrillic_ratio_mean": float(own["cyrillic_ratio"].mean()) if len(own) else 0.0,
        "upper_ratio_mean":    float(own["upper_ratio"].mean())    if len(own) else 0.0,
        "has_newline_ratio":   float(own["has_newline"].mean())    if len(own) else 0.0,
        "ends_with_q_ratio":   float(own["ends_with_q"].mean())    if len(own) else 0.0,
        "has_url_ratio":       float(own["has_url"].mean())        if len(own) else 0.0,
        "char_len_diff_opp":   float(np.mean(ol) - opp["char_len"].mean()) if len(opp) else 0.0,
        "n_utt_diff_opp":      len(own) - len(opp),
    }


def aggregate(dialogs, flat, labels):
    rows = []
    for _, row in labels.iterrows():
        did  = str(row["dialog_id"])
        pidx = int(row["participant_index"])
        if did not in dialogs:
            continue
        fd = flat[flat["dialog_id"] == did]
        feats = {"dialog_id": did, "participant_index": pidx}
        feats.update(participant_stats(fd, pidx))
        rows.append(feats)
    return pd.DataFrame(rows)


def print_summary(stats, labels, name):
    merged = labels.merge(stats, on=["dialog_id","participant_index"], how="left")
    print(f"\n{'='*58}")
    print(f"  {name.upper()} — диалогов: {merged['dialog_id'].nunique()}, участников: {len(merged)}")
    print(f"{'='*58}")

    if "is_bot" in merged.columns:
        b = merged[merged["is_bot"]==1]
        h = merged[merged["is_bot"]==0]
        print(f"  Ботов: {len(b)} ({100*len(b)/len(merged):.1f}%)  |  Людей: {len(h)} ({100*len(h)/len(merged):.1f}%)")
        print()
        COLS = ["n_utterances","char_len_mean","char_len_std","char_len_cv",
                "word_count_mean","vocab_richness","bigram_diversity",
                "self_repeat_ratio","echo_ratio",
                "cyrillic_ratio_mean","has_newline_ratio","ends_with_q_ratio"]
        COLS = [c for c in COLS if c in merged.columns]
        print(f"  {'Признак':<28} {'BOT':>8} {'HUMAN':>8} {'Δ':>8}")
        print(f"  {'-'*56}")
        for col in COLS:
            bv = b[col].mean(); hv = h[col].mean()
            print(f"  {col:<28} {bv:>8.3f} {hv:>8.3f} {bv-hv:>+8.3f}")
    else:
        print(merged[["n_utterances","char_len_mean","word_count_mean","vocab_richness"]].describe().to_string())

    # Аномалии
    s1 = merged[merged["n_utterances"] <= 1]
    print(f"\n  Участников с <= 1 репликой: {len(s1)}")
    zero = merged[merged["char_len_mean"] == 0]
    print(f"  Участников с нулевой длиной: {len(zero)}")
    p99 = merged["char_len_mean"].quantile(0.99)
    long_ = merged[merged["char_len_mean"] > p99]
    print(f"  Участников с очень длинными репликами (p99={p99:.0f}): {len(long_)}")


def ascii_hist(values, title, bins=10, width=35):
    arr = np.array([v for v in values if not np.isnan(v)])
    counts, edges = np.histogram(arr, bins=bins)
    m = max(counts)
    print(f"\n  {title}  (n={len(arr)}, mean={arr.mean():.3f})")
    for c, e1, e2 in zip(counts, edges[:-1], edges[1:]):
        bar = chr(0x2588) * int(width * c / m) if m else ""
        print(f"  [{e1:7.2f},{e2:7.2f}) |{bar:<{width}}| {c}")


def main():
    args = parse_args()
    d = Path(args.data_dir)

    print("Загрузка...")
    train_dialogs = json.load(open(d/"train.json"))
    test_dialogs  = json.load(open(d/"test.json"))
    ytrain = pd.read_csv(d/"ytrain.csv")
    ytest  = pd.read_csv(d/"ytest.csv")
    for df in [ytrain, ytest]:
        df["dialog_id"]         = df["dialog_id"].astype(str)
        df["participant_index"] = df["participant_index"].astype(int)

    print("Строим плоские таблицы...")
    train_flat = dialogs_to_flat(train_dialogs)
    test_flat  = dialogs_to_flat(test_dialogs)

    print("Агрегируем train...")
    train_stats = aggregate(train_dialogs, train_flat, ytrain)
    print("Агрегируем test...")
    test_stats  = aggregate(test_dialogs, test_flat, ytest)

    # Сводка
    print_summary(train_stats, ytrain, "train")
    print_summary(test_stats,  ytest,  "test")

    # ASCII гистограммы
    train_full = ytrain.merge(train_stats, on=["dialog_id","participant_index"], how="left")
    if "is_bot" in train_full.columns:
        bots   = train_full[train_full["is_bot"]==1]
        humans = train_full[train_full["is_bot"]==0]
        for col in ["n_utterances","char_len_mean","char_len_cv","vocab_richness","echo_ratio"]:
            ascii_hist(bots[col],   title=f"{col} — BOT")
            ascii_hist(humans[col], title=f"{col} — HUMAN")

    # Корреляции с таргетом
    if "is_bot" in train_full.columns:
        num_cols = train_stats.columns.difference(["dialog_id","participant_index"])
        corr = train_full[list(num_cols)].corrwith(train_full["is_bot"]).sort_values(key=abs, ascending=False)
        print("\n=== Топ-15 корреляций с is_bot ===")
        print(corr.head(15).to_string())

    # Сохраняем
    train_full.to_csv(d/f"train_{args.output}", index=False)
    ytest.merge(test_stats, on=["dialog_id","participant_index"], how="left").to_csv(
        d/f"test_{args.output}", index=False)
    print(f"\nСохранено: train_{args.output}, test_{args.output}")


if __name__ == "__main__":
    main()
