import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from dataset import MODEL_DIR, RESULTS_DIR, build_label_maps, load_data, stratified_split

MAX_LENGTH = 128
BATCH_SIZE = 16


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
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
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def predict(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_DIR}. Run train.py first.")

    df = load_data()
    _, test_df = stratified_split(df)
    label2id, id2label = build_label_maps(df["label"].tolist())
    class_names = [id2label[i] for i in range(len(id2label))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Test set: {len(test_df)} samples")
    print("Test label counts:\n", test_df["label"].value_counts().sort_index())

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    test_labels = test_df["label"].map(label2id).tolist()
    test_dataset = TextClassificationDataset(
        test_df["question"].tolist(), test_labels, tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds, labels = predict(model, test_loader, device)

    accuracy = (preds == labels).mean()
    print(f"\nTest accuracy: {accuracy:.4f}")

    cm = confusion_matrix(labels, preds)
    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, class_names, RESULTS_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
