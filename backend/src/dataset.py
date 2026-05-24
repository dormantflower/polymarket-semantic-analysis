from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BACKEND_DIR / "data" / "polymarket_dataset.csv"
MODEL_DIR = BACKEND_DIR / "models" / "roberta-large-polymarket"
RESULTS_DIR = BACKEND_DIR / "results"

TEST_SIZE = 180
RANDOM_SEED = 42
MODEL_NAME = "roberta-large"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def stratified_split(df: pd.DataFrame, test_size: int = TEST_SIZE, seed: int = RANDOM_SEED):
    """Split with equal labels per split (60 per class when test_size=180)."""
    labels = sorted(df["label"].unique())
    test_per_class = test_size // len(labels)
    if test_size % len(labels) != 0:
        raise ValueError(f"test_size {test_size} must be divisible by {len(labels)} classes")

    train_parts, test_parts = [], []
    for label in labels:
        subset = df[df["label"] == label].sample(frac=1, random_state=seed)
        test_parts.append(subset.iloc[:test_per_class])
        train_parts.append(subset.iloc[test_per_class:])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df


def build_label_maps(labels: list[str]):
    sorted_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(sorted_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label
