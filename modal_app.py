import modal

app = modal.App("you-are-bot-2")

# Образ с зависимостями
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch", "transformers", "sentence-transformers",
        "lightgbm", "scikit-learn", "pandas", "numpy",
        # "openai",           # для Nebius API
        "accelerate", "tqdm",
        "scipy", "joblib",
        "pyarrow",
        "bitsandbytes",
    ])
)

# Персистентное хранилище для данных и артефактов
volume = modal.Volume.from_name("urbot-data", create_if_missing=True)
MOUNT_PATH = "/data"


# ─── Блок 1: Загрузка данных и CPU-признаки ──────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MOUNT_PATH: volume},
    timeout=600,
)
def build_basic_features():
    """
    CPU-шаг: лингвистические + поведенческие признаки.
    Выполняется без GPU, дёшево.
    """
    import json, pandas as pd, numpy as np, warnings
    from scipy.stats import skew


    with open(f"{MOUNT_PATH}/train.json") as f:
        train_dialogs = json.load(f)
    with open(f"{MOUNT_PATH}/test.json") as f:
        test_dialogs = json.load(f)

    ytrain = pd.read_csv(f"{MOUNT_PATH}/ytrain.csv")
    ytest  = pd.read_csv(f"{MOUNT_PATH}/ytest.csv")

    def dialogs_to_df(dialogs):
        rows = []
        for did, msgs in dialogs.items():
            for m in msgs:
                rows.append({
                    "dialog_id": did,
                    "message_idx": m["message"],
                    "participant_index": m["participant_index"],
                    "text": m["text"],
                })
        return pd.DataFrame(rows)

    def utterance_feats(text):
        tokens = text.split()
        cyr = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        return {
            "char_len":       len(text),
            "word_count":     len(tokens),
            "cyrillic_ratio": cyr / max(len(text), 1),
            "unique_char_r":  len(set(text.lower())) / max(len(text), 1),
            "upper_ratio":    sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "has_newline":    int("\n" in text),
            "ends_with_q": int(text.strip().endswith("?")),
            "digit_ratio":    sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            "punct_ratio":    sum(1 for c in ".,!?;:" if c in text) / max(len(text), 1),
        }

    def echo_features(msgs):
        """Эхо-детекция: доля реплик = последней реплике оппонента."""
        prev = {0: None, 1: None}
        counts = {0: {"echo": 0, "total": 0}, 1: {"echo": 0, "total": 0}}
        for m in sorted(msgs, key=lambda x: x["message"]):
            p, opp = m["participant_index"], 1 - m["participant_index"]
            t = m["text"].strip()
            counts[p]["total"] += 1
            if prev[opp] and (t == prev[opp] or t in prev[opp] or prev[opp] in t):
                counts[p]["echo"] += 1
            prev[p] = t
        return {
            f"p{p}_echo_ratio": counts[p]["echo"] / max(counts[p]["total"], 1)
            for p in [0, 1]
        }

    def repetition_self(msgs, pidx):
        """Доля повторений собственных предыдущих реплик."""
        participant_msgs = sorted(
            [m["text"].strip() for m in msgs if m["participant_index"] == pidx],
            key=lambda x: x
        )
        if len(participant_msgs) < 2:
            return 0.0
        repeats = sum(1 for i in range(1, len(participant_msgs))
                      if participant_msgs[i] == participant_msgs[i-1])
        return repeats / max(len(participant_msgs) - 1, 1)

    def safe_skew(vals):
        if len(vals) <= 2:
            return 0.0
        if np.std(vals) < 1e-10:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = skew(vals)
        return float(result) if np.isfinite(result) else 0.0

    def aggregate_stats(group, numeric_cols):
        feats = {}
        for col in numeric_cols:
            vals = group[col].dropna().values
            if len(vals):
                feats[f"{col}_mean"] = float(np.mean(vals))
                feats[f"{col}_std"] = float(np.std(vals))
                feats[f"{col}_median"] = float(np.median(vals))
                feats[f"{col}_skew"] = safe_skew(vals)
                feats[f"{col}_min"] = float(np.min(vals))
                feats[f"{col}_max"] = float(np.max(vals))
                feats[f"{col}_is_constant"] = int(np.std(vals) < 1e-10)
            else:
                for s in ["mean","std","median","skew","min","max"]:
                    feats[f"{col}_{s}"] = 0.0
        return feats

    def ngram_diversity(msgs, pidx, n=2):
        """Доля уникальных n-грамм от общего числа — чем ниже, тем монотоннее."""
        texts = [m["text"].strip().lower() for m in msgs if m["participant_index"] == pidx]
        all_ngrams = []
        for t in texts:
            tokens = t.split()
            all_ngrams += [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not all_ngrams:
            return 1.0
        return len(set(all_ngrams)) / len(all_ngrams)

    def vocab_richness(msgs, pidx):
        """Type-Token Ratio — уникальные слова / все слова."""
        tokens = []
        for m in msgs:
            if m["participant_index"] == pidx:
                tokens += m["text"].strip().lower().split()
        if not tokens:
            return 1.0
        return len(set(tokens)) / len(tokens)

    def structural_uniformity(msgs, pidx):
        """Коэффициент вариации длин реплик — низкий у бота."""
        lengths = [len(m["text"]) for m in msgs if m["participant_index"] == pidx]
        if len(lengths) < 2:
            return 0.0
        mean = np.mean(lengths)
        if mean < 1e-6:
            return 0.0
        return float(np.std(lengths) / mean)  # CV длин — у бота низкий

    def punct_entropy(msgs, pidx):
        """Энтропия пунктуационных символов — бот использует одни и те же."""
        from collections import Counter
        import math
        puncts = []
        for m in msgs:
            if m["participant_index"] == pidx:
                puncts += [c for c in m["text"] if c in ".,!?;:-—"]
        if not puncts:
            return 0.0
        counts = Counter(puncts)
        total = sum(counts.values())
        return float(-sum((v / total) * math.log2(v / total + 1e-9) for v in counts.values()))

    NUMERIC_COLS = ["char_len", "word_count", "cyrillic_ratio",
                    "unique_char_r", "upper_ratio", "has_newline", "ends_with_q",
                    "digit_ratio", "punct_ratio"]

    def process_split(dialogs, labels_df):
        df = dialogs_to_df(dialogs)
        feat_rows = df["text"].apply(utterance_feats).apply(pd.Series)
        df = pd.concat([df, feat_rows], axis=1)

        rows = []
        for _, label_row in labels_df.iterrows():
            did  = label_row["dialog_id"]
            pidx = label_row["participant_index"]
            if did not in dialogs:
                continue

            msgs = dialogs[did]
            group = df[(df["dialog_id"] == did) &
                       (df["participant_index"] == pidx)]

            feats = {"dialog_id": did, "participant_index": pidx}
            feats.update(aggregate_stats(group, NUMERIC_COLS))
            feats["n_utterances"] = len(group)
            feats["self_repeat"] = repetition_self(msgs, pidx)
            feats["bigram_diversity"] = ngram_diversity(msgs, pidx, n=2)
            feats["trigram_diversity"] = ngram_diversity(msgs, pidx, n=3)
            feats["vocab_richness"] = vocab_richness(msgs, pidx)
            feats["len_cv"] = structural_uniformity(msgs, pidx)
            feats["punct_entropy"] = punct_entropy(msgs, pidx)

            # Эхо-признаки (добавляем для нужного участника)
            echo = echo_features(msgs)
            feats[f"echo_ratio"] = echo[f"p{pidx}_echo_ratio"]

            # Inter-participant контраст
            opp_group = df[(df["dialog_id"] == did) &
                           (df["participant_index"] == (1 - pidx))]
            feats["n_utt_diff_opp"] = len(group) - len(opp_group)
            feats["char_len_mean_diff"] = (
                feats["char_len_mean"] - opp_group["char_len"].mean()
                if len(opp_group) else 0.0
            )
            # feats["char_len_std_diff"] = (
            #     feats["char_len_std"] - opp_group["char_len"].std()
            #     if len(opp_group) else 0.0
            # )
            # Не взлетело - дало ухудшение на 0.013
            feats["char_len_std_diff"] = (
                feats["char_len_std"] - opp_group["char_len"].std(ddof=0)   # ddof=0 — смещенная дисперсия
            #     # (population std), никогда не возвращает NaN даже для одного элемента (будет 0.0).
            #     if len(opp_group) else 0.0
            )

            if len(opp_group) >= 2:
                opp_lengths = opp_group["char_len"].values
                self_lengths = group["char_len"].values
                feats["len_ratio_median"] = float(np.median(self_lengths) / (np.median(opp_lengths) + 1e-6))
                feats["len_skew_diff"] = float(safe_skew(self_lengths) - safe_skew(opp_lengths))
                feats["wordcount_ratio"] = float(group["word_count"].mean() / (opp_group["word_count"].mean() + 1e-6))
            else:
                feats["len_ratio_median"] = 1.0
                feats["len_skew_diff"] = 0.0
                feats["wordcount_ratio"] = 1.0
            rows.append(feats)

        return pd.DataFrame(rows)

    train_basic = process_split(train_dialogs, ytrain)
    test_basic  = process_split(test_dialogs, ytest)

    train_basic.to_parquet(f"{MOUNT_PATH}/train_basic.parquet", index=False)
    test_basic.to_parquet(f"{MOUNT_PATH}/test_basic.parquet", index=False)
    volume.commit()
    print(f"Basic features: train {train_basic.shape}, test {test_basic.shape}")

# ─── Блок 2: GPU- семантические эмбеддинги + OOD ─────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",           # достаточно для LaBSE/E5
    volumes={MOUNT_PATH: volume},
    timeout=1800,
)
def build_embedding_features():
    """
    GPU-шаг: LaBSE эмбеддинги для каждого участника,
    затем OOD-score как дополнительный признак.
    """
    import json, pandas as pd, numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.metrics.pairwise import cosine_similarity

    # MODEL_NAME = "sentence-transformers/LaBSE"
    MODEL_NAME = "intfloat/multilingual-e5-large"
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    with open(f"{MOUNT_PATH}/train.json") as f:
        train_dialogs = json.load(f)
    with open(f"{MOUNT_PATH}/test.json") as f:
        test_dialogs = json.load(f)

    ytrain = pd.read_csv(f"{MOUNT_PATH}/ytrain.csv")
    ytest  = pd.read_csv(f"{MOUNT_PATH}/ytest.csv")

    def get_participant_text(dialogs, did, pidx, sep=" [SEP] "):
        """Конкатенирует все реплики участника в один текст."""
        msgs = sorted(
            [m for m in dialogs[did] if m["participant_index"] == pidx],
            key=lambda x: x["message"]
        )
        return sep.join(m["text"] for m in msgs)

    def context_drift(dialogs, did, pidx):
        """
        Косинусное сходство первой и последней реплики.
        Низкое значение = контекстуальная деградация (LLM-сигнал).
        """
        msgs = sorted(
            [m for m in dialogs[did] if m["participant_index"] == pidx],
            key=lambda x: x["message"]
        )
        if len(msgs) < 2:
            return 1.0
        embs = model.encode([msgs[0]["text"], msgs[-1]["text"]])
        return float(cosine_similarity([embs[0]], [embs[1]])[0][0])

    def get_features(dialogs, labels_df):
        full_texts, drifts = [], []
        keys = []

        for _, row in labels_df.iterrows():
            did, pidx = row["dialog_id"], row["participant_index"]
            if did not in dialogs:
                continue
            keys.append((did, pidx))
            full_texts.append(get_participant_text(dialogs, did, pidx))
            drifts.append(context_drift(dialogs, did, pidx))

        # (только для E5, убрать для LaBSE)
        full_texts = ["query: " + t for t in full_texts]

        # Батчевое кодирование
        embeddings = model.encode(full_texts, batch_size=64,
                                  show_progress_bar=True,
                                  normalize_embeddings=True)
        return keys, embeddings, drifts

    train_keys, train_embs, train_drifts = get_features(train_dialogs, ytrain)
    test_keys,  test_embs,  test_drifts  = get_features(test_dialogs,  ytest)

    # OOD-score: Mahalanobis distance от центра "человеческих" эмбеддингов
    # (Архитектура 2 как признак внутри Архитектуры 1)
    human_mask = [
        ytrain.loc[
            (ytrain["dialog_id"] == k[0]) &
            (ytrain["participant_index"] == k[1]), "is_bot"
        ].values[0] == 0
        for k in train_keys
    ]
    human_embs = train_embs[human_mask]

    cov = EmpiricalCovariance().fit(human_embs)
    train_ood = cov.mahalanobis(train_embs)
    test_ood  = cov.mahalanobis(test_embs)

    # Сохраняем всё
    train_emb_df = pd.DataFrame(
        [{"dialog_id": k[0], "participant_index": k[1],
          "context_drift": d, "ood_score": s}
         for k, d, s in zip(train_keys, train_drifts, train_ood)]
    )
    test_emb_df = pd.DataFrame(
        [{"dialog_id": k[0], "participant_index": k[1],
          "context_drift": d, "ood_score": s}
         for k, d, s in zip(test_keys, test_drifts, test_ood)]
    )
    np.save(f"{MOUNT_PATH}/train_embeddings.npy", train_embs)
    np.save(f"{MOUNT_PATH}/test_embeddings.npy",  test_embs)
    train_emb_df.to_parquet(f"{MOUNT_PATH}/train_emb_feats.parquet", index=False)
    test_emb_df.to_parquet(f"{MOUNT_PATH}/test_emb_feats.parquet",   index=False)
    volume.commit()
    print(f"Embeddings: train {train_embs.shape}, test {test_embs.shape}")

# ─── Блок 3: Perplexity через Nebius API ───────────────────────────────────────────────
# @app.function(
#     image=image,
#     volumes={MOUNT_PATH: volume},
#     secrets=[modal.Secret.from_name("nebius-api-key")],
#     timeout=3600,
#     retries=3,
# )
@app.function(
    image=image,
    gpu="A10G",           # ← добавить GPU
    volumes={MOUNT_PATH: volume},
    timeout=3600,         # retries убрать — при ошибке GPU незачем ретраить
)


def build_perplexity_features():
    """
    Вычисление среднего logprob реплик через Nebius API.
    Nebius поддерживает logprobs в /chat/completions (OpenAI-совместимый).
    """
    import json, pandas as pd, numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    MODEL_NAME = "Qwen/Qwen2-7B"
    # MODEL_NAME = "IlyaGusev/saiga_llama3_8b"
    # MODEL_NAME = "ai-forever/rugpt3large_based_on_gpt2"
    # MODEL_NAME = "Qwen/Qwen3-8B-Base"
    # MODEL_NAME = "mistralai/Mistral-7B-v0.3"
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer.padding_side = "left"  # Llama-3 использует left-padding для батча

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token   # проверка для Mistral,
    #     # для Qwen pad_token уже установлен (<|endoftext|>)

    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     dtype=torch.float16,
    # ).to("cuda")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.eval()

    # Sanity check
    # test_ids = tokenizer("Привет", return_tensors="pt").to("cuda")
    test_ids = tokenizer("Привет, как дела?", return_tensors="pt").to("cuda")
    with torch.no_grad():
        test_loss = model(**test_ids, labels=test_ids["input_ids"]).loss
    print(f"Sanity check — logprob('Привет, как дела?'): {-test_loss.item():.4f}")

    def avg_logprob(text: str) -> float | None:
        if len(text.strip()) < 5:
            return None
        text = text.strip()[:300]
        try:
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=128,
            ).to("cuda")
            with torch.no_grad():
                loss = model(**inputs, labels=inputs["input_ids"]).loss
            return float(-loss.item())
        except Exception as e:
            print(f"Error: {e}")
            return None

    # ─── Binoculars: загружаем observer-модель ───
    from transformers import AutoTokenizer as AT2, AutoModelForCausalLM as AM2

    # OBSERVER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
    # OBSERVER_NAME = "ai-forever/rugpt3large_based_on_gpt2"
    # OBSERVER_NAME = "microsoft/DialoGPT-small"
    OBSERVER_NAME = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading observer {OBSERVER_NAME}...")
    obs_tok = AT2.from_pretrained(OBSERVER_NAME)

    if obs_tok.pad_token is None:   # для microsoft/DialoGPT-small
        obs_tok.pad_token = obs_tok.eos_token

    # obs_model = AM2.from_pretrained(OBSERVER_NAME, dtype=torch.float16).to("cuda")
    obs_model = AM2.from_pretrained(
        OBSERVER_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    obs_model.eval()

    def process_dialogs(dialogs, labels_df):
        import torch.nn.functional as F

        # Шаг 1: собираем все тексты
        all_texts = []
        for _, label_row in labels_df.iterrows():
            did, pidx = str(label_row["dialog_id"]), int(label_row["participant_index"])
            if did not in dialogs:
                continue
            msgs = sorted(
                [m for m in dialogs[did] if m["participant_index"] == pidx],
                key=lambda x: x["message"]
            )
            for m in msgs:
                # t = m["text"].strip()[:300] # для qwen и rugpt3large
                # # t = m["text"].strip()       # для llama-3 (контекст 8192 токенов)
                # if len(t) >= 5:
                #     # all_texts.append((did, pidx, t))
                #     all_texts.append((str(did), int(pidx), t))
                t = m["text"].strip()[:400]
                # Фильтруем по числу токенов, а не символов
                tok_len = len(tokenizer.encode(t, add_special_tokens=False))
                if tok_len >= 4:  # минимум 4 реальных токена
                    all_texts.append((str(did), int(pidx), t))

        # Шаг 2: батчевый GPU-inference
        BATCH_SIZE = 16
        logprobs_map = {}
        bino_map = {}

        for i in range(0, len(all_texts), BATCH_SIZE):
            batch = all_texts[i:i + BATCH_SIZE]
            texts_batch = [t for _, _, t in batch]

            # ── Проход 1: перформер → сохраняем logits на CPU ──
            enc = tokenizer(
                texts_batch, return_tensors="pt",
                padding=True, truncation=True, max_length=128,
            ).to("cuda")
            pad_id = tokenizer.pad_token_id or 0

            with torch.no_grad():
                logits_p = model(**enc).logits  # (B, T, V) на GPU

            shift_logits_p = logits_p[:, :-1, :].cpu()  # ← сразу на CPU
            shift_labels_p = enc["input_ids"][:, 1:].cpu()
            del enc, logits_p
            torch.cuda.empty_cache()  # ← освобождаем GPU до наблюдателя

            # ── Проход 2: наблюдатель ──
            enc_obs = obs_tok(
                texts_batch, return_tensors="pt",
                padding=True, truncation=True, max_length=128,
            ).to("cuda")
            pad_id_obs = obs_tok.pad_token_id or 0

            with torch.no_grad():
                logits_o = obs_model(**enc_obs).logits

            shift_logits_o = logits_o[:, :-1, :].cpu()  # ← тоже на CPU
            shift_labels_o = enc_obs["input_ids"][:, 1:].cpu()
            del enc_obs, logits_o
            torch.cuda.empty_cache()

            # ── Единый цикл по репликам (всё на CPU) ──
            for j, (did, pidx, _) in enumerate(batch):
                lb_p = shift_labels_p[j]
                lb_o = shift_labels_o[j]
                mask_p = lb_p != pad_id
                mask_o = lb_o != pad_id_obs
                common_mask = mask_p & mask_o

                if common_mask.sum() == 0:
                    continue

                n_toks = int(common_mask.sum())
                smoothing = max(0.0, 0.1 * (1 - n_toks / 8)) if n_toks < 8 else 0.0

                loss_p = F.cross_entropy(
                    shift_logits_p[j][common_mask].float(),
                    lb_p[common_mask],
                    label_smoothing=smoothing,
                )
                loss_o = F.cross_entropy(
                    shift_logits_o[j][common_mask].float(),
                    lb_o[common_mask],
                    label_smoothing=smoothing,
                )

                lp_perf = float(-loss_p.item())
                lp_obs = float(-loss_o.item())

                logprobs_map.setdefault((str(did), int(pidx)), []).append(lp_perf)
                bino = lp_perf / (lp_obs - 1e-8) if abs(lp_obs) > 1e-6 else float('nan')
                bino_map.setdefault((str(did), int(pidx)), []).append(bino)

            del shift_logits_p, shift_labels_p, shift_logits_o, shift_labels_o

            if i % (BATCH_SIZE * 50) == 0:
                print(f"  logprob batch {i // BATCH_SIZE}/{len(all_texts) // BATCH_SIZE}")

        # Шаг 3: агрегация
        rows = []
        for _, label_row in labels_df.iterrows():
            # did, pidx = label_row["dialog_id"], label_row["participant_index"]
            # if did not in dialogs:
            #     continue
            # lp_clean = logprobs_map.get((did, pidx), [])
            did, pidx = str(label_row["dialog_id"]), int(label_row["participant_index"])
            if did not in dialogs:
                continue
            lp_clean = logprobs_map.get((did, pidx), [])
            bino_vals = bino_map.get((did, pidx), [])
            # bino_clean = [v for v in bino_vals if not np.isnan(v)]
            rows.append({
                "dialog_id": did,
                "participant_index": pidx,
                "logprob_mean": np.mean(lp_clean) if lp_clean else 0.0,
                "logprob_std": np.std(lp_clean) if len(lp_clean) > 1 else 0.0,
                "logprob_min": np.min(lp_clean) if lp_clean else 0.0,
                "logprob_max": float(np.max(lp_clean)) if lp_clean else 0.0,
                # "bino_mean": np.mean(bino_vals) if bino_vals else 1.0,  # ← ДОБАВИТЬ
                # "bino_std": np.std(bino_vals) if len(bino_vals) > 1 else 0.0,  # ← ДОБАВИТЬ
                "bino_mean": float(np.nanmean(bino_vals)) if bino_vals else 0.0,            # лучший результат
                "bino_std": float(np.nanstd(bino_vals)) if len(bino_vals) > 1 else 0.0,
                # "bino_mean": float(np.mean(bino_clean)) if bino_clean else 0.0,           # не взлетело
                # "bino_std": float(np.std(bino_clean)) if len(bino_clean) > 1 else 0.0,
                "bino_max": float(np.nanmax(bino_vals)) if bino_vals else 0.0,
                "bino_min": float(np.nanmin(bino_vals)) if bino_vals else 0.0,
            })
        return pd.DataFrame(rows)

    with open(f"{MOUNT_PATH}/train.json") as f:
        train_dialogs = json.load(f)
    with open(f"{MOUNT_PATH}/test.json") as f:
        test_dialogs = json.load(f)

    ytrain = pd.read_csv(f"{MOUNT_PATH}/ytrain.csv")
    ytest  = pd.read_csv(f"{MOUNT_PATH}/ytest.csv")

    train_lp = process_dialogs(train_dialogs, ytrain)
    test_lp  = process_dialogs(test_dialogs,  ytest)

    train_lp.to_parquet(f"{MOUNT_PATH}/train_logprob.parquet", index=False)
    test_lp.to_parquet(f"{MOUNT_PATH}/test_logprob.parquet",  index=False)
    volume.commit()
    print(f"Logprob features done: train {len(train_lp)}, test {len(test_lp)}")

# ─── Блок 4: Мета-классификатор. TF-IDF мета-признак + финальный LightGBM ────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MOUNT_PATH: volume},
    timeout=1200,
)
def train_and_predict():
    """
    Сборка всех признаков, OOF TF-IDF, LightGBM + калибровка.
    Метрика: LogLoss → нужна хорошая калибровка вероятностей.
    """
    import json, pandas as pd, numpy as np
    import lightgbm as lgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import log_loss

    from sklearn.model_selection import train_test_split

    from sklearn.decomposition import PCA

    import pandas as pd

    # Загрузка данных
    with open(f"{MOUNT_PATH}/train.json") as f:
        train_dialogs = json.load(f)
    with open(f"{MOUNT_PATH}/test.json") as f:
        test_dialogs = json.load(f)

    ytrain = pd.read_csv(f"{MOUNT_PATH}/ytrain.csv")
    ytest  = pd.read_csv(f"{MOUNT_PATH}/ytest.csv")

    # Загрузка признаков
    train_basic = pd.read_parquet(f"{MOUNT_PATH}/train_basic.parquet")
    test_basic  = pd.read_parquet(f"{MOUNT_PATH}/test_basic.parquet")
    train_emb   = pd.read_parquet(f"{MOUNT_PATH}/train_emb_feats.parquet")
    test_emb    = pd.read_parquet(f"{MOUNT_PATH}/test_emb_feats.parquet")
    train_lp    = pd.read_parquet(f"{MOUNT_PATH}/train_logprob.parquet")
    test_lp     = pd.read_parquet(f"{MOUNT_PATH}/test_logprob.parquet")

    # Заменяем нулевой дефолт на медиану по train
    bino_median = train_lp["bino_mean"].median()
    logprob_median = train_lp["logprob_mean"].median()
    train_lp["bino_mean"] = train_lp["bino_mean"].replace(0.0, bino_median)
    test_lp["bino_mean"] = test_lp["bino_mean"].replace(0.0, bino_median)
    train_lp["logprob_mean"] = train_lp["logprob_mean"].replace(0.0, logprob_median)
    test_lp["logprob_mean"] = test_lp["logprob_mean"].replace(0.0, logprob_median)

    # приведение типов ключей везде
    for df in [ytrain, ytest, train_basic, test_basic,
               train_emb, test_emb, train_lp, test_lp]:
        df["dialog_id"] = df["dialog_id"].astype(str)
        df["participant_index"] = df["participant_index"].astype(int)

    KEY_COLS = ["dialog_id", "participant_index"]

    print("logprob train sample:")
    print(train_lp[["logprob_mean", "logprob_std", "logprob_min"]].describe())
    print("NaN count:", train_lp[["logprob_mean"]].isna().sum().values)

    def merge_all(labels, *frames):
        df = labels.copy()
        for frame in frames:
            df = df.merge(frame, on=KEY_COLS, how="left")
        return df

    # TF-IDF на первых 2 репликах каждого участника (OOF для train)
    def get_early_text(dialogs, did, pidx, n=2):
        msgs = sorted(
            [m for m in dialogs.get(did, []) if m["participant_index"] == pidx],
            key=lambda x: x["message"]
        )
        return " ".join(m["text"] for m in msgs[:n])

    early_train = [get_early_text(train_dialogs, r.dialog_id, r.participant_index)
                   for _, r in ytrain.iterrows()]
    early_test  = [get_early_text(test_dialogs,  r.dialog_id, r.participant_index)
                   for _, r in ytest.iterrows()]

    tfidf = TfidfVectorizer(max_features=8000, analyzer="char_wb",
                            ngram_range=(2, 4))
    X_tfidf_tr = tfidf.fit_transform(early_train)
    X_tfidf_te = tfidf.transform(early_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_train = ytrain["is_bot"].values

    lr = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
    tfidf_oof = cross_val_predict(lr, X_tfidf_tr, y_train,
                                  cv=skf, method="predict_proba")[:, 1]
    lr.fit(X_tfidf_tr, y_train)
    tfidf_test_pred = lr.predict_proba(X_tfidf_te)[:, 1]

    ytrain["tfidf_lr"] = tfidf_oof
    ytest["tfidf_lr"]  = tfidf_test_pred

    # ─── Отдельные TF-IDF модели на N-й реплике (сильный сигнал по writeup) ───
    def get_nth_utterance(dialogs, did, pidx, n=0):
        """Возвращает n-ю реплику участника (0-индексация)."""
        msgs = sorted(
            [m for m in dialogs.get(str(did), []) if m["participant_index"] == int(pidx)],
            key=lambda x: x["message"]
        )
        return msgs[n]["text"] if len(msgs) > n else ""

    for utt_idx, col_name in [(0, "tfidf_utt0"), (1, "tfidf_utt1")]:
        texts_tr = [get_nth_utterance(train_dialogs, r.dialog_id, r.participant_index, utt_idx)
                    for _, r in ytrain.iterrows()]
        texts_te = [get_nth_utterance(test_dialogs, r.dialog_id, r.participant_index, utt_idx)
                    for _, r in ytest.iterrows()]

        # Фильтруем пустые строки: LR не обучится на пустом тексте
        non_empty_mask = [len(t.strip()) > 0 for t in texts_tr]
        print(f"utt{utt_idx}: non-empty train={sum(non_empty_mask)}/{len(texts_tr)}")

        tfidf_n = TfidfVectorizer(max_features=5000, analyzer="char_wb",
                                  ngram_range=(2, 4), min_df=2)
        X_n_tr = tfidf_n.fit_transform(texts_tr)
        X_n_te = tfidf_n.transform(texts_te)

        lr_n = LogisticRegression(C=0.3, max_iter=1000, solver="lbfgs")
        oof_n = cross_val_predict(lr_n, X_n_tr, y_train,
                                  cv=skf, method="predict_proba")[:, 1]
        lr_n.fit(X_n_tr, y_train)

        oof_ll = log_loss(y_train, oof_n)
        print(f"tfidf_utt{utt_idx} OOF logloss: {oof_ll:.4f}")

        ytrain[col_name] = oof_n
        ytest[col_name] = lr_n.predict_proba(X_n_te)[:, 1]

    # Финальная матрица
    meta_train = merge_all(ytrain, train_basic, train_emb, train_lp)
    meta_test  = merge_all(ytest,  test_basic,  test_emb,  test_lp)

    meta_train["logprob_cv"] = meta_train["logprob_std"] / (meta_train["logprob_mean"].abs() + 1e-8)
    meta_test["logprob_cv"] = meta_test["logprob_std"] / (meta_test["logprob_mean"].abs() + 1e-8)
    # Coefficient of variation — нормализованная дисперсия

    meta_train["bino_cv"] = meta_train["bino_std"] / (meta_train["bino_mean"].abs() + 1e-8)
    meta_test["bino_cv"] = meta_test["bino_std"] / (meta_test["bino_mean"].abs() + 1e-8)
    # bino_cv — коэффициент вариации bino по репликам:
    # у бота реплики однородны → bino_cv низкий; у человека — разнородны

    EXCLUDE = {
        "dialog_id", "participant_index", "is_bot", "ID", "ood_score",
        # cyrillic_ratio — корреляция 0.045, только шум
        "cyrillic_ratio_mean", "cyrillic_ratio_std", "cyrillic_ratio_median",
        "cyrillic_ratio_skew", "cyrillic_ratio_min", "cyrillic_ratio_max",
        "cyrillic_ratio_is_constant",
        "char_len_is_constant", "upper_ratio_is_constant", "word_count_is_constant",
        "digit_ratio_median", "digit_ratio_mean", "unique_char_r_is_constant",
        "has_newline_skew", "trigram_diversity", "ends_with_q_max", "ends_with_q_min",
        "digit_ratio_min", "digit_ratio_is_constant"
    }
    # оставляем "ood_score"
    # EXCLUDE = {
    #     "dialog_id", "participant_index", "is_bot", "ID",
    #     # cyrillic_ratio — корреляция 0.045, только шум
    #     "cyrillic_ratio_mean", "cyrillic_ratio_std", "cyrillic_ratio_median",
    #     "cyrillic_ratio_skew", "cyrillic_ratio_min", "cyrillic_ratio_max",
    #     "cyrillic_ratio_is_constant",
    #     "char_len_is_constant", "upper_ratio_is_constant", "word_count_is_constant",
    #     "digit_ratio_median", "digit_ratio_mean", "unique_char_r_is_constant",
    #     "has_newline_skew", "trigram_diversity", "ends_with_q_max", "ends_with_q_min",
    #     "digit_ratio_min", "digit_ratio_is_constant"
    # }
    FEAT_COLS = [c for c in meta_train.columns if c not in EXCLUDE]

    # Диагностика
    print("FEAT_COLS:", FEAT_COLS)
    print("logprob_mean variance:", meta_train["logprob_mean"].var())
    print("logprob_mean unique:", meta_train["logprob_mean"].nunique())
    print("NaN in meta_train:", meta_train[FEAT_COLS].isna().sum().sum())

    nan_cols = meta_train[FEAT_COLS].isna().sum()
    print("NaN per column:\n", nan_cols[nan_cols > 0])

    # ─── Диагностика: train vs test distribution shift ───
    print("\n=== Distribution shift: train vs test ===")
    shift_report = []
    for col in FEAT_COLS:
        tr_mean = meta_train[col].mean()
        te_mean = meta_test[col].mean()
        tr_std = meta_train[col].std() + 1e-9
        shift = abs(tr_mean - te_mean) / tr_std  # normalized shift
        shift_report.append({"feature": col, "train_mean": tr_mean,
                             "test_mean": te_mean, "shift_sigma": shift})

    shift_df = pd.DataFrame(shift_report).sort_values("shift_sigma", ascending=False)
    print(shift_df[shift_df["shift_sigma"] > 0.3].to_string())  # только подозрительные

    # ─── Шаг 2а: предварительная модель для отбора признаков ───
    lgb_pre = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, random_state=42,
        colsample_bytree=0.7, subsample=0.8,
    )
    lgb_pre.fit(meta_train[FEAT_COLS].fillna(0), y_train)

    importance_df = pd.DataFrame({
        "feature": FEAT_COLS,
        "importance": lgb_pre.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nFeature importances (top-20):")
    print(importance_df.head(20).to_string())

    zero_feats = set(importance_df[importance_df["importance"] == 0]["feature"].tolist())
    print(f"\nRemoving zero-importance: {len(zero_feats)} features: {zero_feats}")

    FEAT_COLS = [c for c in FEAT_COLS if c not in zero_feats]  # ← перезаписываем FEAT_COLS
    print(f"Features after filtering: {len(FEAT_COLS)}")

    train_embs_raw = np.load(f"{MOUNT_PATH}/train_embeddings.npy")  # (3032, 768)  → (3032, 1024)
    test_embs_raw = np.load(f"{MOUNT_PATH}/test_embeddings.npy")  # (758, 768)  → (758, 1024)

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    def add_cross_sim(labels_df, embs_raw):
        """Косинусное сходство эмбеддингов p0 и p1 одного диалога."""
        # Строим словарь (dialog_id, pidx) → индекс строки в embs_raw
        key_to_idx = {}
        for i, (_, row) in enumerate(labels_df.iterrows()):
            key_to_idx[(str(row["dialog_id"]), int(row["participant_index"]))] = i

        sims = []
        for _, row in labels_df.iterrows():
            did = str(row["dialog_id"])
            pidx = int(row["participant_index"])
            opp = 1 - pidx
            i_self = key_to_idx.get((did, pidx))
            i_opp = key_to_idx.get((did, opp))
            if i_self is not None and i_opp is not None:
                s = float(cos_sim([embs_raw[i_self]], [embs_raw[i_opp]])[0][0])
            else:
                s = 0.0
            sims.append(s)
        return sims

    meta_train["cross_sim"] = add_cross_sim(meta_train, train_embs_raw)
    meta_test["cross_sim"] = add_cross_sim(meta_test, test_embs_raw)

    # Добавить "cross_sim" в FEAT_COLS — он не попадёт в EXCLUDE,
    # поэтому подхватится автоматически при переопределении FEAT_COLS:
    FEAT_COLS = [c for c in meta_train.columns if c not in EXCLUDE]
    # ↑ ВАЖНО: эту строку перенести ПОСЛЕ добавления cross_sim в meta_train


    medians = meta_train[FEAT_COLS].median()

    # pca = PCA(n_components=64, random_state=42)
    pca = PCA(n_components=0.90, random_state=42)  # объяснить 90% дисперсии
    train_embs_pca = pca.fit_transform(train_embs_raw)
    test_embs_pca = pca.transform(test_embs_raw)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    X_tr = np.hstack([meta_train[FEAT_COLS].fillna(medians).values, train_embs_pca])
    X_te = np.hstack([meta_test[FEAT_COLS].fillna(medians).values, test_embs_pca])
    # X_tr = np.hstack([meta_train[FEAT_COLS].fillna(medians).values, train_embs_raw])
    # X_te = np.hstack([meta_test[FEAT_COLS].fillna(medians).values, test_embs_raw])
    y_tr = meta_train["is_bot"].values

    # Диагностика после fix
    print(f"meta_train shape after fix: {meta_train.shape}")  # должно быть (3032, ...)
    print(f"NaN after fillna: {np.isnan(X_tr).sum()}")  # должно быть 0

    # LightGBM с ранней остановкой
    # lgb_model = lgb.LGBMClassifier(
    #     n_estimators=1000,
    #     learning_rate=0.03,
    #     num_leaves=31, # было 63
    #     verbosity=-1,
    #     subsample=0.8,
    #     colsample_bytree=0.7,
    #     min_child_samples=10, # было 20
    #     reg_alpha=0.1,
    #     reg_lambda=0.1,
    #     random_state=42,
    #     # class_weight="balanced",
    #     # is_unbalance=True,
    # )
    #
    # # OOF для оценки LogLoss
    # oof_preds = cross_val_predict(lgb_model, X_tr, y_tr,
    #                               cv=skf, method="predict_proba")[:, 1]
    # print(f"OOF LogLoss: {log_loss(y_tr, oof_preds):.4f}")
    #
    # # Калибровка вероятностей (КРИТИЧНО для LogLoss!)
    # # CalibratedClassifierCV с method='isotonic' для нелинейной коррекции
    # # calibrated = CalibratedClassifierCV(lgb_model, method="isotonic", cv=5)
    #
    # # calibrated = CalibratedClassifierCV(lgb_model, method="sigmoid", cv=5)
    # # calibrated.fit(X_tr, y_tr)
    # #
    # # test_probs = calibrated.predict_proba(X_te)[:, 1]
    # # test_probs = np.clip(test_probs, 0.001, 0.999)
    #
    # lgb_model.fit(X_tr, y_tr)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        verbosity=-1,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
    )

    # OOF оценка (без early stopping — для честного сравнения)
    oof_preds = cross_val_predict(lgb_model, X_tr, y_tr,
                                  cv=skf, method="predict_proba")[:, 1]
    print(f"OOF LogLoss: {log_loss(y_tr, oof_preds):.4f}")

    # Финальное обучение с early stopping на hold-out 10%
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42
    )
    lgb_model_final = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=31,
        verbosity=-1,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
    )
    lgb_model_final.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)]
    )

    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import cross_val_predict

    oof_cal = cross_val_predict(
        lgb_model_final, X_tr, y_tr,
        cv=skf, method="predict_proba"
    )[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_cal, y_tr)


    print(f"Best iteration: {lgb_model_final.best_iteration_}")

    # ===============================================
    # INFO: Важность признаков
    feat_names = list(FEAT_COLS) + [f"pca_{i}" for i in range(train_embs_pca.shape[1])]
    importances = lgb_model_final.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("=== TOP 20 признаков ===")
    print(importance_df.head(20).to_string())
    print("\n=== НУЛЕВЫЕ признаки ===")
    zero_feats = importance_df[importance_df["importance"] == 0]
    print(f"Признаков с нулевой важностью: {len(zero_feats)}")
    print(zero_feats["feature"].tolist())
    # ===============================================

    # # test_probs = lgb_model.predict_proba(X_te)[:, 1]
    # test_probs = lgb_model_final.predict_proba(X_te)[:, 1]
    # test_probs = np.clip(test_probs, 0.001, 0.999)

    raw_probs = lgb_model_final.predict_proba(X_te)[:, 1]
    test_probs = iso.predict(raw_probs)
    test_probs = np.clip(test_probs, 0.001, 0.999)  # оставить — isotonic может дать 0.0/1.0

    # Формирование сабмиссии в нужном формате
    submission = ytest[["ID"]].copy()
    submission["is_bot"] = test_probs
    submission.to_csv(f"{MOUNT_PATH}/submission_2.csv", index=False)
    volume.commit()
    print(f"Submission saved: {len(submission)} rows")
    print(submission.head())

# ─── Блок 5: Оркестрация, запуск всего пайплайна ───────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """
    Последовательный запуск всех этапов.
    CPU-шаги выполняются параллельно с GPU-шагами через .spawn().
    """
    import modal

    print("Step 1: Basic features (CPU)...")
    build_basic_features.remote()

    print("Step 2: Embeddings + OOD score (GPU)...")
    embed_handle = build_embedding_features.spawn()   # параллельно

    print("Step 3: Perplexity via Qwen...")
    perp_handle = build_perplexity_features.spawn()   # параллельно

    # Ждём завершения параллельных шагов
    embed_handle.get()
    perp_handle.get()

    print("Step 4: Train meta-learner + calibrate + submit...")
    train_and_predict.remote()

    volume_obj = modal.Volume.from_name("urbot-data")
    with open("submission_3.csv", "wb") as f:
        for chunk in volume_obj.read_file("submission_3.csv"):
            f.write(chunk)
    print("Downloaded: submission_2.csv")

    print("Pipeline complete! Submission at /data/submission_2.csv")