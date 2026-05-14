#!/usr/bin/env python3
"""
analyze_dataset.py -- статистический анализ диалогов train/test.
Запуск: python analyze_dataset.py --data_dir /path/to/data

Признаки:
  УДАЛЕНЫ (нулевые/слабые): cyrillic_ratio_mean, self_repeat_ratio, total_chars,
      total_words, n_opp_utterances, has_url_ratio, char_len_mean (оставлен только
      для информации, не в сводной таблице)
  ДОБАВЛЕНЫ НОВЫЕ: n_utt_diff_opp, has_newline_ratio, ends_with_q_ratio,
      char_len_skew, char_len_is_constant, word_count_skew, word_count_is_constant,
      unique_char_r_*, digit_ratio_*, punct_ratio_*, punct_entropy,
      len_skew_diff, char_len_std_diff, len_ratio_median, wordcount_ratio,
      logprob_cv (из parquet если доступен)

  ПУЛ БУДУЩИХ признаков (требуют GPU / внешних моделей):
      context_drift, logprob_mean/std/min, bino_mean/std, logprob_cv
      -- добавляются автоматически если найдены *_logprob.parquet и *_emb_feats.parquet
"""

import json
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from scipy.stats import skew as scipy_skew


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".", help="Папка с train/test json и csv")
    parser.add_argument("--output",   default="dataset_analysis.csv")
    return parser.parse_args()


# ─── Уттеранс-уровень ────────────────────────────────────────────────────────

def dialogs_to_flat(dialogs):
    rows = []
    for did, msgs in dialogs.items():
        for m in msgs:
            text   = m["text"]
            tokens = text.split()
            rows.append({
                "dialog_id":         did,
                "participant_index": m["participant_index"],
                "message_idx":       m["message"],
                "char_len":          len(text),
                "word_count":        len(tokens),
                "unique_char_r":     len(set(text.lower())) / max(len(text), 1),
                "upper_ratio":       sum(1 for c in text if c.isupper()) / max(len(text), 1),
                "digit_ratio":       sum(1 for c in text if c.isdigit()) / max(len(text), 1),
                "punct_ratio":       sum(1 for c in ".,!?;:" if c in text) / max(len(text), 1),
                "has_newline":       int("\n" in text),
                "ends_with_q":       int(text.strip().endswith("?")),
                "text":              text,
            })
    return pd.DataFrame(rows)


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def safe_skew(vals):
    if len(vals) <= 2 or np.std(vals) < 1e-10:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = scipy_skew(vals)
    return float(r) if np.isfinite(r) else 0.0


def agg6(vals):
    """mean, std, median, skew, min, max, is_constant для массива."""
    if len(vals) == 0:
        return {s: 0.0 for s in ["mean","std","median","skew","min","max","is_constant"]}
    return {
        "mean":        float(np.mean(vals)),
        "std":         float(np.std(vals)),
        "median":      float(np.median(vals)),
        "skew":        safe_skew(vals),
        "min":         float(np.min(vals)),
        "max":         float(np.max(vals)),
        "is_constant": int(np.std(vals) < 1e-10),
    }


def ngram_diversity(texts, n=2):
    all_ngrams = []
    for t in texts:
        toks = t.lower().split()
        all_ngrams += [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return len(set(all_ngrams)) / max(len(all_ngrams), 1)


def vocab_richness(texts):
    tokens = [w for t in texts for w in t.lower().split()]
    return len(set(tokens)) / max(len(tokens), 1)


def echo_ratio_fn(own_texts, opp_texts):
    """Доля реплик, содержащих / содержащихся в предыдущей реплике оппонента."""
    if not opp_texts or not own_texts:
        return 0.0
    echoes = 0
    for i, t in enumerate(own_texts):
        prev_opp = opp_texts[i-1] if 0 < i <= len(opp_texts) else opp_texts[0]
        if prev_opp and (t == prev_opp or t in prev_opp or prev_opp in t):
            echoes += 1
    return echoes / len(own_texts)


def punct_entropy_fn(texts):
    """Энтропия распределения пунктуационных символов."""
    puncts = [c for t in texts for c in t if c in ".,!?;:-—"]
    if not puncts:
        return 0.0
    cnt = Counter(puncts)
    total = sum(cnt.values())
    return float(-sum((v/total) * math.log2(v/total + 1e-9) for v in cnt.values()))


def structural_uniformity(texts):
    """CV длин реплик — низкий у бота (они одинаково длинные)."""
    lengths = [len(t) for t in texts]
    if len(lengths) < 2:
        return 0.0
    mean = np.mean(lengths)
    return float(np.std(lengths) / (mean + 1e-6))


# ─── Per-participant агрегация ────────────────────────────────────────────────

def participant_stats(flat_dialog, pidx):
    own  = flat_dialog[flat_dialog["participant_index"] == pidx].sort_values("message_idx")
    opp  = flat_dialog[flat_dialog["participant_index"] == (1 - pidx)]
    ot   = own["text"].tolist()
    op   = opp["text"].tolist()
    ol   = own["char_len"].values
    wl   = own["word_count"].values
    ucr  = own["unique_char_r"].values
    ur   = own["upper_ratio"].values
    dr   = own["digit_ratio"].values
    pr   = own["punct_ratio"].values

    feats = {
        # ── Базовые счётчики ──────────────────────────────────────────────
        "n_utterances":      len(own),
        "n_utt_diff_opp":    len(own) - len(opp),   # r=0.62 — сильнейший признак

        # ── Длина реплик (все 6 агрегатов — зеркало modal_app.py) ────────
        **{f"char_len_{k}": v for k, v in agg6(ol).items()},

        # ── Число слов ───────────────────────────────────────────────────
        **{f"word_count_{k}": v for k, v in agg6(wl).items()},

        # ── Уникальные символы ───────────────────────────────────────────
        **{f"unique_char_r_{k}": v for k, v in agg6(ucr).items()},

        # ── Заглавные буквы ──────────────────────────────────────────────
        **{f"upper_ratio_{k}": v for k, v in agg6(ur).items()},

        # ── Цифры ────────────────────────────────────────────────────────
        **{f"digit_ratio_{k}": v for k, v in agg6(dr).items()},

        # ── Пунктуация ───────────────────────────────────────────────────
        **{f"punct_ratio_{k}": v for k, v in agg6(pr).items()},

        # ── Лексика ──────────────────────────────────────────────────────
        "vocab_richness":    vocab_richness(ot),
        "bigram_diversity":  ngram_diversity(ot, 2),
        "trigram_diversity": ngram_diversity(ot, 3),

        # ── Повторения ───────────────────────────────────────────────────
        "echo_ratio":        echo_ratio_fn(ot, op),  # r=0.35 — 2-й по силе

        # ── Структурная однородность ─────────────────────────────────────
        "len_cv":            structural_uniformity(ot),
        "punct_entropy":     punct_entropy_fn(ot),

        # ── Форматирование (сильные новые признаки) ───────────────────────
        "has_newline_ratio": float(own["has_newline"].mean()) if len(own) else 0.0,
        "ends_with_q_ratio": float(own["ends_with_q"].mean()) if len(own) else 0.0,

        # ── Контраст с оппонентом ────────────────────────────────────────
        "char_len_mean_diff":  float(np.mean(ol) - opp["char_len"].mean()) if len(opp) else 0.0,
        "char_len_std_diff":   float(np.std(ol) - opp["char_len"].std())   if len(opp) > 1 else 0.0,
        "len_ratio_median":    float(np.median(ol) / (np.median(opp["char_len"].values) + 1e-6)) if len(opp) >= 2 else 1.0,
        "len_skew_diff":       float(safe_skew(ol) - safe_skew(opp["char_len"].values)) if len(opp) >= 2 else 0.0,
        "wordcount_ratio":     float(np.mean(wl) / (opp["word_count"].mean() + 1e-6)) if len(opp) else 1.0,

        # ── Флаг однорепличного участника ────────────────────────────────
        "is_single_utt":     int(len(own) <= 1),
    }
    return feats


# ─── Агрегация по всем диалогам ──────────────────────────────────────────────

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


# ─── Подключение GPU-признаков из parquet (если есть) ────────────────────────

def try_merge_gpu_feats(stats_df, d, split):
    """
    Пытается подключить context_drift, ood_score, logprob_*, bino_*
    из файлов, уже посчитанных modal_app.py.
    """
    emb_path = d / f"{split}_emb_feats.parquet"
    lp_path  = d / f"{split}_logprob.parquet"
    merged   = stats_df.copy()

    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        emb["dialog_id"]         = emb["dialog_id"].astype(str)
        emb["participant_index"] = emb["participant_index"].astype(int)
        merged = merged.merge(emb, on=["dialog_id","participant_index"], how="left")
        print(f"  + Подключены эмбеддинг-признаки ({emb_path.name}): {list(emb.columns[2:])}")

    if lp_path.exists():
        lp = pd.read_parquet(lp_path)
        lp["dialog_id"]         = lp["dialog_id"].astype(str)
        lp["participant_index"] = lp["participant_index"].astype(int)
        # logprob_cv = std / |mean|  (CV перплексии)
        if "logprob_mean" in lp.columns and "logprob_std" in lp.columns:
            lp["logprob_cv"] = lp["logprob_std"] / (lp["logprob_mean"].abs() + 1e-6)
        merged = merged.merge(lp, on=["dialog_id","participant_index"], how="left")
        print(f"  + Подключены logprob-признаки ({lp_path.name}): {list(lp.columns[2:])}")

    return merged


# ─── Сводный вывод ───────────────────────────────────────────────────────────

# Признаки для таблицы bot vs human (только информативные)
SUMMARY_COLS = [
    "n_utterances", "n_utt_diff_opp",
    "char_len_mean", "char_len_std", "char_len_cv",
    "word_count_mean", "word_count_std",
    "vocab_richness", "bigram_diversity", "trigram_diversity",
    "echo_ratio", "len_cv", "punct_entropy",
    "has_newline_ratio", "ends_with_q_ratio",
    "char_len_mean_diff", "len_ratio_median", "wordcount_ratio",
    # GPU-признаки добавятся автоматически если есть
    "context_drift", "logprob_mean", "logprob_cv", "bino_mean",
]

# Признаки для ASCII гистограмм
HIST_COLS = [
    "n_utt_diff_opp", "echo_ratio", "has_newline_ratio",
    "char_len_std", "char_len_cv", "vocab_richness", "word_count_mean",
]


def print_summary(merged, name):
    print(f"\n{'='*62}")
    print(f"  {name.upper()} — диалогов: {merged['dialog_id'].nunique()}, участников: {len(merged)}")
    print(f"{'='*62}")

    if "is_bot" in merged.columns:
        b = merged[merged["is_bot"] == 1]
        h = merged[merged["is_bot"] == 0]
        print(f"  Ботов: {len(b)} ({100*len(b)/len(merged):.1f}%)  |"
              f"  Людей: {len(h)} ({100*len(h)/len(merged):.1f}%)\n")

        cols = [c for c in SUMMARY_COLS if c in merged.columns]
        print(f"  {'Признак':<28} {'BOT':>9} {'HUMAN':>9} {'Delta':>9}")
        print(f"  {'-'*59}")
        for col in cols:
            bv = b[col].mean(); hv = h[col].mean()
            marker = " <" if abs(bv - hv) > 0.05 * max(abs(bv), abs(hv), 1) else ""
            print(f"  {col:<28} {bv:>9.3f} {hv:>9.3f} {bv-hv:>+9.3f}{marker}")
    else:
        desc_cols = ["n_utterances", "char_len_mean", "word_count_mean", "vocab_richness", "echo_ratio"]
        desc_cols = [c for c in desc_cols if c in merged.columns]
        print(merged[desc_cols].describe().round(3).to_string())

    # Аномалии
    s1    = merged[merged["n_utterances"] <= 1]
    p99   = merged["char_len_mean"].quantile(0.99)
    long_ = merged[merged["char_len_mean"] > p99]
    print(f"\n  Участников с <=1 репликой (is_single_utt): {len(s1)}")
    print(f"  Участников с очень длинными репликами (p99={p99:.0f}): {len(long_)}")


def ascii_hist(values, title, bins=10, width=35):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return
    counts, edges = np.histogram(arr, bins=bins)
    m = max(counts)
    print(f"\n  {title}  (n={len(arr)}, mean={arr.mean():.3f}, std={arr.std():.3f})")
    for c, e1, e2 in zip(counts, edges[:-1], edges[1:]):
        bar = chr(0x2588) * int(width * c / m) if m else ""
        print(f"  [{e1:8.3f},{e2:8.3f}) |{bar:<{width}}| {c}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    d = Path(args.data_dir)

    print("Загрузка данных...")
    train_dialogs = json.load(open(d / "train.json"))
    test_dialogs  = json.load(open(d / "test.json"))
    ytrain = pd.read_csv(d / "ytrain.csv")
    ytest  = pd.read_csv(d / "ytest.csv")
    for df in [ytrain, ytest]:
        df["dialog_id"]         = df["dialog_id"].astype(str)
        df["participant_index"] = df["participant_index"].astype(int)

    print("Уттеранс-уровень...")
    train_flat = dialogs_to_flat(train_dialogs)
    test_flat  = dialogs_to_flat(test_dialogs)

    print("Агрегируем train...")
    train_stats = aggregate(train_dialogs, train_flat, ytrain)
    print("Агрегируем test...")
    test_stats  = aggregate(test_dialogs, test_flat, ytest)

    # Подключаем GPU-признаки если уже посчитаны modal_app.py
    print("\nПоиск GPU-признаков (logprob / embeddings)...")
    train_stats = try_merge_gpu_feats(train_stats, d, "train")
    test_stats  = try_merge_gpu_feats(test_stats,  d, "test")

    # Присоединяем метки
    train_full = ytrain.merge(train_stats, on=["dialog_id","participant_index"], how="left")
    test_full  = ytest.merge(test_stats,   on=["dialog_id","participant_index"], how="left")

    # Сводки
    print_summary(train_full, "train")
    print_summary(test_full,  "test")

    # ASCII гистограммы (только train)
    if "is_bot" in train_full.columns:
        bots   = train_full[train_full["is_bot"] == 1]
        humans = train_full[train_full["is_bot"] == 0]
        extra_hist = [c for c in ["logprob_mean","bino_mean","context_drift"] if c in train_full.columns]
        for col in HIST_COLS + extra_hist:
            if col in train_full.columns:
                ascii_hist(bots[col],   title=f"{col} — BOT")
                ascii_hist(humans[col], title=f"{col} — HUMAN")

    # Корреляции с таргетом
    if "is_bot" in train_full.columns:
        num_cols = [c for c in train_full.columns
                    if c not in ("dialog_id","participant_index","is_bot")
                    and pd.api.types.is_numeric_dtype(train_full[c])]
        corr = train_full[num_cols].corrwith(train_full["is_bot"]).sort_values(key=abs, ascending=False)
        print("\n=== Top-20 корреляций с is_bot ===")
        print(corr.head(20).to_string())

    # Сохраняем CSV
    train_full.to_csv(d / f"train_{args.output}", index=False)
    test_full.to_csv(d  / f"test_{args.output}",  index=False)
    print(f"\nСохранено: train_{args.output}, test_{args.output}")

    # =========================================================================
    # ПУЛ БУДУЩИХ ПРИЗНАКОВ (не реализованы, требуют GPU / внешних библиотек)
    # =========================================================================
    print("""
+--------------------------------------------------------------+
|  ПУЛ ПРИЗНАКОВ ДЛЯ БУДУЩЕГО ВКЛЮЧЕНИЯ                       |
+--------------------------------------------------------------+
|  GPU / трансформеры:                                         |
|    context_drift_window3  -- косинус окном 3 реплики         |
|      (сейчас только first vs last)                           |
|    bm25_echo  -- BM25-сходство реплики с предыдущей          |
|      репликой оппонента (мягче чем substring-echo)           |
|    ner_ratio  -- доля реплик с именованными сущностями        |
|      (spacy ru_core_news_sm)                                 |
|                                                              |
|  Топик-моделирование (CPU):                                  |
|    topic_diversity  -- кол-во SVD-компонент > порога          |
|      (TfidfVectorizer + TruncatedSVD на репликах участника)  |
|    topic_entropy    -- энтропия распределения тем            |
|                                                              |
|  Временные (если есть timestamps):                           |
|    msg_gap_mean / msg_gap_std  -- среднее/СКО времени        |
|      между репликами участника                               |
|    response_time_ratio  -- скорость ответа vs оппонент       |
|                                                              |
|  Калибровка:                                                 |
|    isotonic_calibration  -- Isotonic Regression по OOF       |
|      предсказаниям LightGBM (ожидаемый эффект -0.003..-0.008)|
+--------------------------------------------------------------+
""")


if __name__ == "__main__":
    main()
