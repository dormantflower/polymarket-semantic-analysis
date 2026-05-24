import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

from dataset import (
    MODEL_DIR,
    MODEL_NAME,
    build_label_maps,
    load_data,
    stratified_split,
)

MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


def main():
    df = load_data()
    train_df, test_df = stratified_split(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print("Train label counts:\n", train_df["label"].value_counts().sort_index())
    print("Test label counts:\n", test_df["label"].value_counts().sort_index())

    label2id, id2label = build_label_maps(df["label"].tolist())
    num_labels = len(label2id)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    train_labels = train_df["label"].map(label2id).tolist()
    test_labels = test_df["label"].map(label2id).tolist()

    train_dataset = TextClassificationDataset(
        train_df["question"].tolist(), train_labels, tokenizer
    )
    eval_dataset = TextClassificationDataset(
        test_df["question"].tolist(), test_labels, tokenizer
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=25,
        logging_first_step=True,
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining {MODEL_NAME} for {NUM_EPOCHS} epochs...")
    trainer.train()

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nModel saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
