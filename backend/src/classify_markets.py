import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from dataset import BACKEND_DIR, MODEL_DIR, RESULTS_DIR

MAX_LENGTH = 128
BATCH_SIZE = 64
HF_DATASET = "SII-WANGZJ/Polymarket_data"

MARKETS_CLASSIFIED_PATH = BACKEND_DIR / "data" / "markets_classified.parquet"
MARKETS_SUMMARY_PATH = RESULTS_DIR / "markets_classification_summary.json"
MARKETS_SUMMARY_TXT = RESULTS_DIR / "markets_classification_summary.txt"


def find_markets_parquet() -> Path:
    local = BACKEND_DIR / "data" / "markets.parquet"
    if local.exists():
        return local

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    matches = sorted(
        cache_root.glob("datasets--SII-WANGZJ--Polymarket_data/snapshots/*/markets.parquet")
    )
    if matches:
        return matches[-1]

    from datasets import load_dataset

    cache_dir = BACKEND_DIR / "data" / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        HF_DATASET,
        data_files=["markets.parquet"],
        cache_dir=str(cache_dir),
    )
    return Path(dataset["train"]._data.files[0])  # noqa: SLF001


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


@torch.inference_mode()
def classify_texts(model, texts, tokenizer, device, batch_size=BATCH_SIZE):
    dataset = InferenceDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    use_fp16 = device.type == "cuda"
    total_batches = len(loader)

    all_preds, all_confs = [], []
    pbar = tqdm(loader, total=total_batches, desc="Classifying markets", unit="batch")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits.float(), dim=-1)
        confs, preds = probs.max(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())
        pbar.set_postfix(markets=f"{len(all_preds):,}/{len(texts):,}")

    return np.array(all_preds), np.array(all_confs)


def build_summary(df: pd.DataFrame, id2label: dict, elapsed_s: float) -> dict:
    counts = df["predicted_label"].value_counts()
    total = len(df)
    summary = {
        "total_markets_classified": total,
        "elapsed_seconds": round(elapsed_s, 2),
        "label_distribution": {
            label: {
                "count": int(counts.get(label, 0)),
                "percent": round(100 * counts.get(label, 0) / total, 2),
            }
            for label in id2label.values()
        },
        "mean_confidence": round(float(df["confidence"].mean()), 4),
        "mean_confidence_by_label": {
            label: round(float(df.loc[df["predicted_label"] == label, "confidence"].mean()), 4)
            for label in id2label.values()
            if label in df["predicted_label"].values
        },
        "low_confidence_count_below_0.5": int((df["confidence"] < 0.5).sum()),
    }
    return summary


def save_summary(summary: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MARKETS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    lines = [
        "Markets classification summary",
        "=" * 40,
        f"Markets classified: {summary['total_markets_classified']:,}",
        f"Elapsed: {summary['elapsed_seconds']:.1f}s",
        f"Mean confidence: {summary['mean_confidence']:.4f}",
        f"Low confidence (<0.5): {summary['low_confidence_count_below_0.5']:,}",
        "",
        "Predicted label distribution:",
    ]
    for label, stats in summary["label_distribution"].items():
        lines.append(f"  {label}: {stats['count']:,} ({stats['percent']}%)")
    lines.append("")
    lines.append("Mean confidence by label:")
    for label, conf in summary["mean_confidence_by_label"].items():
        lines.append(f"  {label}: {conf:.4f}")

    MARKETS_SUMMARY_TXT.write_text("\n".join(lines) + "\n")


def main():
    import time

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_DIR}. Run train.py first.")

    parquet_path = find_markets_parquet()
    tqdm.write(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    tqdm.write(f"Loaded {len(df):,} markets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}")

    tqdm.write("Loading model...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    texts = df["question"].fillna("").astype(str).tolist()
    start = time.perf_counter()
    pred_ids, confidences = classify_texts(model, texts, tokenizer, device)
    elapsed = time.perf_counter() - start
    tqdm.write(f"Classification done in {elapsed:.1f}s ({len(texts) / elapsed:,.0f} markets/s)")

    df["predicted_label"] = [id2label[int(i)] for i in pred_ids]
    df["confidence"] = confidences

    MARKETS_CLASSIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Saving to {MARKETS_CLASSIFIED_PATH}...")
    df.to_parquet(MARKETS_CLASSIFIED_PATH, index=False)
    tqdm.write("Save complete.")

    summary = build_summary(df, id2label, elapsed)
    save_summary(summary)
    tqdm.write(f"Saved summary to {MARKETS_SUMMARY_PATH}")
    tqdm.write("\n" + MARKETS_SUMMARY_TXT.read_text())


if __name__ == "__main__":
    main()
