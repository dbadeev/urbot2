import modal

app = modal.App("you-are-bot-2")

# Образ с зависимостями
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch", "transformers", "sentence-transformers",
        "lightgbm", "scikit-learn", "pandas", "numpy",
        # "openai",           # Для Nebius API
        "accelerate", "tqdm",
        "scipy", "joblib",
        "pyarrow",
    ])
)

# Персистентное хранилище для данных и артефактов
volume = modal.Volume.from_name("urbot-data", create_if_missing=True)
MOUNT_PATH = "/data"


# в”Ђв”Ђв”Ђ Р‘Р»РѕРє 1: Р—Р°РіСЂСѓР·РєР° РґР°РЅРЅС‹С… Рё CPU-РїСЂРёР·РЅР°РєРё в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.function(
    image=image,
    volumes={MOUNT_PATH: volume},
    timeout=600,
)
def build_basic_features():
    """
    CPU-С€Р°Рі: Р»РёРЅРіРІРёСЃС‚РёС‡РµСЃРєРёРµ + РїРѕРІРµРґРµРЅС‡РµСЃРєРёРµ РїСЂРёР·РЅР°РєРё.
    Р’С‹РїРѕР»РЅСЏРµС‚СЃСЏ Р±РµР· GPU, РґС‘С€РµРІРѕ.
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
            "digit_ratio":    sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            "punct_ratio":    sum(1 for c in ".,!?;:" if c in text) / max(len(text), 1),
        }

    def echo_features(msgs):
        """Р­С…Рѕ-РґРµС‚РµРєС†РёСЏ: РґРѕР»СЏ СЂРµРїР»РёРє = РїРѕСЃР»РµРґРЅРµР№ СЂРµРїР»РёРєРµ РѕРїРїРѕРЅРµРЅС‚Р°."""
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
        """Р”РѕР»СЏ РїРѕРІС‚РѕСЂРµРЅРёР№ СЃРѕР±СЃС‚РІРµРЅРЅС‹С… РїСЂРµРґС‹РґСѓС‰РёС… СЂРµРїР»РёРє."""
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

    NUMERIC_COLS = ["char_len","word_count","cyrillic_ratio",
                    "unique_char_r","upper_ratio","digit_ratio","punct_ratio"]

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

            # Р­С…Рѕ-РїСЂРёР·РЅР°РєРё (РґРѕР±Р°РІР»СЏРµРј РґР»СЏ РЅСѓР¶РЅРѕРіРѕ СѓС‡Р°СЃС‚РЅРёРєР°)
            echo = echo_features(msgs)
            feats[f"echo_ratio"] = echo[f"p{pidx}_echo_ratio"]

            # Inter-participant РєРѕРЅС‚СЂР°СЃС‚
            opp_group = df[(df["dialog_id"] == did) &
                           (df["participant_index"] == (1 - pidx))]
            feats["char_len_mean_diff"] = (
                feats["char_len_mean"] - opp_group["char_len"].mean()
                if len(opp_group) else 0.0
            )
            feats["char_len_std_diff"] = (
                feats["char_len_std"] - opp_group["char_len"].std()
                if len(opp_group) else 0.0
            )
            rows.append(feats)

        return pd.DataFrame(rows)

    train_basic = process_split(train_dialogs, ytrain)
    test_basic  = process_split(test_dialogs, ytest)

    train_basic.to_parquet(f"{MOUNT_PATH}/train_basic.parquet", index=False)
    test_basic.to_parquet(f"{MOUNT_PATH}/test_basic.parquet", index=False)
    volume.commit()
    print(f"Basic features: train {train_basic.shape}, test {test_basic.shape}")

# в”Ђв”Ђв”Ђ Р‘Р»РѕРє 2: GPU- СЃРµРјР°РЅС‚РёС‡РµСЃРєРёРµ СЌРјР±РµРґРґРёРЅРіРё + OOD в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.function(
    image=image,
    gpu="A10G",           # РґРѕСЃС‚Р°С‚РѕС‡РЅРѕ РґР»СЏ LaBSE/E5
    volumes={MOUNT_PATH: volume},
    timeout=1800,
)
def build_embedding_features():
    """
    GPU-С€Р°Рі: LaBSE СЌРјР±РµРґРґРёРЅРіРё РґР»СЏ РєР°Р¶РґРѕРіРѕ СѓС‡Р°СЃС‚РЅРёРєР°,
    Р·Р°С‚РµРј OOD-score РєР°Рє РґРѕРїРѕР»РЅРёС‚РµР»СЊРЅС‹Р№ РїСЂРёР·РЅР°Рє.
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
        """РљРѕРЅРєР°С‚РµРЅРёСЂСѓРµС‚ РІСЃРµ СЂРµРїР»РёРєРё СѓС‡Р°СЃС‚РЅРёРєР° РІ РѕРґРёРЅ С‚РµРєСЃС‚."""
        msgs = sorted(
            [m for m in dialogs[did] if m["participant_index"] == pidx],
            key=lambda x: x["message"]
        )
        return sep.join(m["text"] for m in msgs)

    def context_drift(dialogs, did, pidx):
        """
        РљРѕСЃРёРЅСѓСЃРЅРѕРµ СЃС…РѕРґСЃС‚РІРѕ РїРµСЂРІРѕР№ Рё РїРѕСЃР»РµРґРЅРµР№ СЂРµРїР»РёРєРё.
        РќРёР·РєРѕРµ Р·РЅР°С‡РµРЅРёРµ = РєРѕРЅС‚РµРєСЃС‚СѓР°Р»СЊРЅР°СЏ РґРµРіСЂР°РґР°С†РёСЏ (LLM-СЃРёРіРЅР°Р»).
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

        # (С‚РѕР»СЊРєРѕ РґР»СЏ E5, СѓР±СЂР°С‚СЊ РґР»СЏ LaBSE)
        full_texts = ["query: " + t for t in full_texts]

        # Р‘Р°С‚С‡РµРІРѕРµ РєРѕРґРёСЂРѕРІР°РЅРёРµ
        embeddings = model.encode(full_texts, batch_size=64,
                                  show_progress_bar=True,
                                  normalize_embeddings=True)
        return keys, embeddings, drifts

    train_keys, train_embs, train_drifts = get_features(train_dialogs, ytrain)
    test_keys,  test_embs,  test_drifts  = get_features(test_dialogs,  ytest)

    # OOD-score: Mahalanobis distance РѕС‚ С†РµРЅС‚СЂР° "С‡РµР»РѕРІРµС‡РµСЃРєРёС…" СЌРјР±РµРґРґРёРЅРіРѕРІ
    # (РђСЂС…РёС‚РµРєС‚СѓСЂР° 2 РєР°Рє РїСЂРёР·РЅР°Рє РІРЅСѓС‚СЂРё РђСЂС…РёС‚РµРєС‚СѓСЂС‹ 1)
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

    # РЎРѕС…СЂР°РЅСЏРµРј РІСЃС‘
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

# в”Ђв”Ђв”Ђ Р‘Р»РѕРє 3: Perplexity С‡РµСЂРµР· Nebius API в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# @app.function(
#     image=image,
#     volumes={MOUNT_PATH: volume},
#     secrets=[modal.Secret.from_name("nebius-api-key")],
#     timeout=3600,
#     retries=3,
# )
@app.function(
    image=image,
    gpu="A10G",           # в†ђ РґРѕР±Р°РІРёС‚СЊ GPU
    volumes={MOUNT_PATH: volume},
    timeout=3600,         # retries СѓР±СЂР°С‚СЊ вЂ” РїСЂРё РѕС€РёР±РєРµ GPU РЅРµР·Р°С‡РµРј СЂРµС‚СЂР°РёС‚СЊ
)


def build_perplexity_features():
    """
    Р’С‹С‡РёСЃР»РµРЅРёРµ СЃСЂРµРґРЅРµРіРѕ logprob СЂРµРїР»РёРє С‡РµСЂРµР· Nebius API.
    Nebius РїРѕРґРґРµСЂР¶РёРІР°РµС‚ logprobs РІ /chat/completions (OpenAI-СЃРѕРІРјРµСЃС‚РёРјС‹Р№).
    """
    import json, pandas as pd, numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_NAME = "Qwen/Qwen2-7B"
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()

    # Sanity check
    test_ids = tokenizer("РџСЂРёРІРµС‚", return_tensors="pt").to("cuda")
    with torch.no_grad():
        test_loss = model(**test_ids, labels=test_ids["input_ids"]).loss
    print(f"Sanity check вЂ” logprob('РџСЂРёРІРµС‚'): {-test_loss.item():.4f}")

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

    # в”Ђв”Ђв”Ђ Binoculars: Р·Р°РіСЂСѓР¶Р°РµРј observer-РјРѕРґРµР»СЊ в”Ђв”Ђв”Ђ
    from transformers import AutoTokenizer as AT2, AutoModelForCausalLM as AM2

    OBSERVER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
    print(f"Loading observer {OBSERVER_NAME}...")
    obs_tok = AT2.from_pretrained(OBSERVER_NAME)
    obs_model = AM2.from_pretrained(OBSERVER_NAME, dtype=torch.float16).to("cuda")
    obs_model.eval()

    def process_dialogs(dialogs, labels_df):
        import torch.nn.functional as F

        # РЁР°Рі 1: СЃРѕР±РёСЂР°РµРј РІСЃРµ С‚РµРєСЃС‚С‹
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
                t = m["text"].strip()[:300]
                if len(t) >= 5:
                    # all_texts.append((did, pidx, t))
                    all_texts.append((str(did), int(pidx), t))

        # РЁР°Рі 2: Р±Р°С‚С‡РµРІС‹Р№ GPU-inference
        BATCH_SIZE = 16
        logprobs_map = {}
        bino_map = {}

        for i in range(0, len(all_texts), BATCH_SIZE):
            batch = all_texts[i:i + BATCH_SIZE]
            texts_batch = [t for _, _, t in batch]

            enc = tokenizer(
                texts_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to("cuda")

            with torch.no_grad():
                logits = model(**enc).logits  # (B, T, V)

            shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
            shift_labels = enc["input_ids"][:, 1:].contiguous()   # (B, T-1)
            pad_id = tokenizer.pad_token_id or 0

            batch_lp = {}

            for j, (did, pidx, _) in enumerate(batch):
                lb = shift_labels[j]
                mask = lb != pad_id
                if mask.sum() == 0:
                    continue
                    # cross_entropy РІ fp16 вЂ” Р±РµР· РєРѕРЅРІРµСЂС‚Р°С†РёРё РІ fp32
                loss = F.cross_entropy(
                    shift_logits[j][mask].float(),  # С‚РѕР»СЊРєРѕ РЅСѓР¶РЅС‹Р№ СЃСЂРµР·, РЅРµ РІРµСЃСЊ С‚РµРЅР·РѕСЂ
                    lb[mask]
                )

                lp = float(-loss.item())
                logprobs_map.setdefault((str(did), int(pidx)), []).append(lp)
                batch_lp[j] = lp  # в†ђ СЃРѕС…СЂР°РЅСЏС‚СЊ per-j
                # logprobs_map.setdefault((did, pidx), []).append(float(-loss.item()))
                # logprobs_map.setdefault((str(did), int(pidx)), []).append(float(-loss.item()))

            del enc, logits, shift_logits, shift_labels
            torch.cuda.empty_cache()  # в†ђ РѕСЃРІРѕР±РѕР¶РґР°С‚СЊ РїРѕСЃР»Рµ РєР°Р¶РґРѕРіРѕ Р±Р°С‚С‡Р°

            # в”Ђв”Ђ Observer batch (ruGPT-small) РґР»СЏ Binoculars в”Ђв”Ђ
            enc_obs = obs_tok(
                texts_batch,
                return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to("cuda")
            with torch.no_grad():
                logits_obs = obs_model(**enc_obs).logits

            shift_logits_obs = logits_obs[:, :-1, :].contiguous()
            shift_labels_obs = enc_obs["input_ids"][:, 1:].contiguous()
            pad_id_obs = obs_tok.pad_token_id or 0

            for j, (did, pidx, _) in enumerate(batch):
                lb = shift_labels_obs[j]
                mask = lb != pad_id_obs
                if mask.sum() == 0:
                    continue
                loss_obs = F.cross_entropy(
                    shift_logits_obs[j][mask].float(), lb[mask]
                )
                lp_list = logprobs_map.get((str(did), int(pidx)), [])
                # lp_perf = lp_list[-1] if lp_list else None
                # if lp_perf and lp_perf != 0:
                #     bino = float(-loss_obs.item()) / abs(lp_perf)
                # else:
                #     bino = 1.0
                lp_perf = batch_lp.get(j)  # в†ђ С‚РѕС‡РЅРѕРµ Р·РЅР°С‡РµРЅРёРµ РґР»СЏ СЌС‚РѕР№ СЂРµРїР»РёРєРё
                if lp_perf and lp_perf != 0:
                    # lp_perf < 0 (logprob Qwen), loss_obs > 0 (CE ruGPT-small)
                    # РћР±Р° РІ РѕРґРЅРѕРј Р·РЅР°РєРѕРІРѕРј РїСЂРѕСЃС‚СЂР°РЅСЃС‚РІРµ Р»РѕРіР°СЂРёС„РјРѕРІ:
                    bino = lp_perf / (-float(loss_obs.item()) - 1e-8)
                    # bino = float(-loss_obs.item()) / abs(lp_perf)
                else:
                    bino = float('nan')
                bino_map.setdefault((str(did), int(pidx)), []).append(bino)

            del enc_obs, logits_obs, shift_logits_obs, shift_labels_obs
            torch.cuda.empty_cache()

            if i % (BATCH_SIZE * 50) == 0:
                print(f"  logprob batch {i // BATCH_SIZE}/{len(all_texts) // BATCH_SIZE}")

        # РЁР°Рі 3: Р°РіСЂРµРіР°С†РёСЏ
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
            rows.append({
                "dialog_id": did,
                "participant_index": pidx,
                "logprob_mean": np.mean(lp_clean) if lp_clean else 0.0,
                "logprob_std": np.std(lp_clean) if len(lp_clean) > 1 else 0.0,
                "logprob_min": np.min(lp_clean) if lp_clean else 0.0,
                # "bino_mean": np.mean(bino_vals) if bino_vals else 1.0,  # в†ђ Р”РћР‘РђР’РРўР¬
                # "bino_std": np.std(bino_vals) if len(bino_vals) > 1 else 0.0,  # в†ђ Р”РћР‘РђР’РРўР¬
                "bino_mean": float(np.nanmean(bino_vals)) if bino_vals else 0.0,
                "bino_std": float(np.nanstd(bino_vals)) if len(bino_vals) > 1 else 0.0,
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

# в”Ђв”Ђв”Ђ Р‘Р»РѕРє 4: РњРµС‚Р°-РєР»Р°СЃСЃРёС„РёРєР°С‚РѕСЂ. TF-IDF РјРµС‚Р°-РїСЂРёР·РЅР°Рє + С„РёРЅР°Р»СЊРЅС‹Р№ LightGBM в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.function(
    image=image,
    volumes={MOUNT_PATH: volume},
    timeout=1200,
)
def train_and_predict():
    """
    РЎР±РѕСЂРєР° РІСЃРµС… РїСЂРёР·РЅР°РєРѕРІ, OOF TF-IDF, LightGBM + РєР°Р»РёР±СЂРѕРІРєР°.
    РњРµС‚СЂРёРєР°: LogLoss в†’ РЅСѓР¶РЅР° С…РѕСЂРѕС€Р°СЏ РєР°Р»РёР±СЂРѕРІРєР° РІРµСЂРѕСЏС‚РЅРѕСЃС‚РµР№.
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

    # Р—Р°РіСЂСѓР·РєР° РґР°РЅРЅС‹С…
    with open(f"{MOUNT_PATH}/train.json") as f:
        train_dialogs = json.load(f)
    with open(f"{MOUNT_PATH}/test.json") as f:
        test_dialogs = json.load(f)

    ytrain = pd.read_csv(f"{MOUNT_PATH}/ytrain.csv")
    ytest  = pd.read_csv(f"{MOUNT_PATH}/ytest.csv")

    # Р—Р°РіСЂСѓР·РєР° РїСЂРёР·РЅР°РєРѕРІ
    train_basic = pd.read_parquet(f"{MOUNT_PATH}/train_basic.parquet")
    test_basic  = pd.read_parquet(f"{MOUNT_PATH}/test_basic.parquet")
    train_emb   = pd.read_parquet(f"{MOUNT_PATH}/train_emb_feats.parquet")
    test_emb    = pd.read_parquet(f"{MOUNT_PATH}/test_emb_feats.parquet")
    train_lp    = pd.read_parquet(f"{MOUNT_PATH}/train_logprob.parquet")
    test_lp     = pd.read_parquet(f"{MOUNT_PATH}/test_logprob.parquet")

    # РїСЂРёРІРµРґРµРЅРёРµ С‚РёРїРѕРІ РєР»СЋС‡РµР№ РІРµР·РґРµ
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

    # TF-IDF РЅР° РїРµСЂРІС‹С… 2 СЂРµРїР»РёРєР°С… РєР°Р¶РґРѕРіРѕ СѓС‡Р°СЃС‚РЅРёРєР° (OOF РґР»СЏ train)
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

    # Р¤РёРЅР°Р»СЊРЅР°СЏ РјР°С‚СЂРёС†Р°
    meta_train = merge_all(ytrain, train_basic, train_emb, train_lp)
    meta_test  = merge_all(ytest,  test_basic,  test_emb,  test_lp)

    meta_train["logprob_cv"] = meta_train["logprob_std"] / (meta_train["logprob_mean"].abs() + 1e-8)
    meta_test["logprob_cv"] = meta_test["logprob_std"] / (meta_test["logprob_mean"].abs() + 1e-8)
    # Coefficient of variation вЂ” РЅРѕСЂРјР°Р»РёР·РѕРІР°РЅРЅР°СЏ РґРёСЃРїРµСЂСЃРёСЏ

    # EXCLUDE = {"dialog_id", "participant_index", "is_bot", "ID"}
    EXCLUDE = {"dialog_id", "participant_index", "is_bot", "ID", "ood_score"}
    FEAT_COLS = [c for c in meta_train.columns if c not in EXCLUDE]

    # Р”РёР°РіРЅРѕСЃС‚РёРєР°
    print("FEAT_COLS:", FEAT_COLS)
    print("logprob_mean variance:", meta_train["logprob_mean"].var())
    print("logprob_mean unique:", meta_train["logprob_mean"].nunique())
    print("NaN in meta_train:", meta_train[FEAT_COLS].isna().sum().sum())

    nan_cols = meta_train[FEAT_COLS].isna().sum()
    print("NaN per column:\n", nan_cols[nan_cols > 0])

    train_embs_raw = np.load(f"{MOUNT_PATH}/train_embeddings.npy")  # (3032, 768)  в†’ (3032, 1024)
    test_embs_raw = np.load(f"{MOUNT_PATH}/test_embeddings.npy")  # (758, 768)  в†’ (758, 1024)

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    def add_cross_sim(labels_df, embs_raw):
        """РљРѕСЃРёРЅСѓСЃРЅРѕРµ СЃС…РѕРґСЃС‚РІРѕ СЌРјР±РµРґРґРёРЅРіРѕРІ p0 Рё p1 РѕРґРЅРѕРіРѕ РґРёР°Р»РѕРіР°."""
        # РЎС‚СЂРѕРёРј СЃР»РѕРІР°СЂСЊ (dialog_id, pidx) в†’ РёРЅРґРµРєСЃ СЃС‚СЂРѕРєРё РІ embs_raw
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

    # Р”РѕР±Р°РІРёС‚СЊ "cross_sim" РІ FEAT_COLS вЂ” РѕРЅ РЅРµ РїРѕРїР°РґС‘С‚ РІ EXCLUDE,
    # РїРѕСЌС‚РѕРјСѓ РїРѕРґС…РІР°С‚РёС‚СЃСЏ Р°РІС‚РѕРјР°С‚РёС‡РµСЃРєРё РїСЂРё РїРµСЂРµРѕРїСЂРµРґРµР»РµРЅРёРё FEAT_COLS:
    FEAT_COLS = [c for c in meta_train.columns if c not in EXCLUDE]
    # в†‘ Р’РђР–РќРћ: СЌС‚Сѓ СЃС‚СЂРѕРєСѓ РїРµСЂРµРЅРµСЃС‚Рё РџРћРЎР›Р• РґРѕР±Р°РІР»РµРЅРёСЏ cross_sim РІ meta_train


    medians = meta_train[FEAT_COLS].median()

    # pca = PCA(n_components=64, random_state=42)
    pca = PCA(n_components=0.90, random_state=42)  # РѕР±СЉСЏСЃРЅРёС‚СЊ 90% РґРёСЃРїРµСЂСЃРёРё
    train_embs_pca = pca.fit_transform(train_embs_raw)
    test_embs_pca = pca.transform(test_embs_raw)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    X_tr = np.hstack([meta_train[FEAT_COLS].fillna(medians).values, train_embs_pca])
    X_te = np.hstack([meta_test[FEAT_COLS].fillna(medians).values, test_embs_pca])
    # X_tr = np.hstack([meta_train[FEAT_COLS].fillna(medians).values, train_embs_raw])
    # X_te = np.hstack([meta_test[FEAT_COLS].fillna(medians).values, test_embs_raw])
    y_tr = meta_train["is_bot"].values

    # Р”РёР°РіРЅРѕСЃС‚РёРєР° РїРѕСЃР»Рµ fix
    print(f"meta_train shape after fix: {meta_train.shape}")  # РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ (3032, ...)
    print(f"NaN after fillna: {np.isnan(X_tr).sum()}")  # РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ 0

    # LightGBM СЃ СЂР°РЅРЅРµР№ РѕСЃС‚Р°РЅРѕРІРєРѕР№
    # lgb_model = lgb.LGBMClassifier(
    #     n_estimators=1000,
    #     learning_rate=0.03,
    #     num_leaves=31, # Р±С‹Р»Рѕ 63
    #     verbosity=-1,
    #     subsample=0.8,
    #     colsample_bytree=0.7,
    #     min_child_samples=10, # Р±С‹Р»Рѕ 20
    #     reg_alpha=0.1,
    #     reg_lambda=0.1,
    #     random_state=42,
    #     # class_weight="balanced",
    #     # is_unbalance=True,
    # )
    #
    # # OOF РґР»СЏ РѕС†РµРЅРєРё LogLoss
    # oof_preds = cross_val_predict(lgb_model, X_tr, y_tr,
    #                               cv=skf, method="predict_proba")[:, 1]
    # print(f"OOF LogLoss: {log_loss(y_tr, oof_preds):.4f}")
    #
    # # РљР°Р»РёР±СЂРѕРІРєР° РІРµСЂРѕСЏС‚РЅРѕСЃС‚РµР№ (РљР РРўРР§РќРћ РґР»СЏ LogLoss!)
    # # CalibratedClassifierCV СЃ method='isotonic' РґР»СЏ РЅРµР»РёРЅРµР№РЅРѕР№ РєРѕСЂСЂРµРєС†РёРё
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

    # OOF РѕС†РµРЅРєР° (Р±РµР· early stopping вЂ” РґР»СЏ С‡РµСЃС‚РЅРѕРіРѕ СЃСЂР°РІРЅРµРЅРёСЏ)
    oof_preds = cross_val_predict(lgb_model, X_tr, y_tr,
                                  cv=skf, method="predict_proba")[:, 1]
    print(f"OOF LogLoss: {log_loss(y_tr, oof_preds):.4f}")

    # Р¤РёРЅР°Р»СЊРЅРѕРµ РѕР±СѓС‡РµРЅРёРµ СЃ early stopping РЅР° hold-out 10%
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
    print(f"Best iteration: {lgb_model_final.best_iteration_}")

    # test_probs = lgb_model.predict_proba(X_te)[:, 1]
    test_probs = lgb_model_final.predict_proba(X_te)[:, 1]
    test_probs = np.clip(test_probs, 0.001, 0.999)

    # Р¤РѕСЂРјРёСЂРѕРІР°РЅРёРµ СЃР°Р±РјРёСЃСЃРёРё РІ РЅСѓР¶РЅРѕРј С„РѕСЂРјР°С‚Рµ
    submission = ytest[["ID"]].copy()
    submission["is_bot"] = test_probs
    submission.to_csv(f"{MOUNT_PATH}/submission_2.csv", index=False)
    volume.commit()
    print(f"Submission saved: {len(submission)} rows")
    print(submission.head())

# в”Ђв”Ђв”Ђ Р‘Р»РѕРє 5: РћСЂРєРµСЃС‚СЂР°С†РёСЏ, Р·Р°РїСѓСЃРє РІСЃРµРіРѕ РїР°Р№РїР»Р°Р№РЅР° в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.local_entrypoint()
def main():
    """
    РџРѕСЃР»РµРґРѕРІР°С‚РµР»СЊРЅС‹Р№ Р·Р°РїСѓСЃРє РІСЃРµС… СЌС‚Р°РїРѕРІ.
    CPU-С€Р°РіРё РІС‹РїРѕР»РЅСЏСЋС‚СЃСЏ РїР°СЂР°Р»Р»РµР»СЊРЅРѕ СЃ GPU-С€Р°РіР°РјРё С‡РµСЂРµР· .spawn().
    """
    import modal

    print("Step 1: Basic features (CPU)...")
    build_basic_features.remote()

    print("Step 2: Embeddings + OOD score (GPU)...")
    embed_handle = build_embedding_features.spawn()   # РїР°СЂР°Р»Р»РµР»СЊРЅРѕ

    print("Step 3: Perplexity via Qwen...")
    perp_handle = build_perplexity_features.spawn()   # РїР°СЂР°Р»Р»РµР»СЊРЅРѕ

    # Р–РґС‘Рј Р·Р°РІРµСЂС€РµРЅРёСЏ РїР°СЂР°Р»Р»РµР»СЊРЅС‹С… С€Р°РіРѕРІ
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