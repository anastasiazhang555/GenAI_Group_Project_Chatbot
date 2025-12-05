# train_emotion_model.py

import csv
import re
import pickle
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from app.emotion_model import (
    TextEmotionDataset,
    LSTMEmotionClassifier,
    simple_tokenize,
    DEVICE,
)

# -----------------------------
# 1. Emotion mapping (NEW)
# -----------------------------
# Map EmpatheticDialogues "context" strings → 4 emotion categories
# anger=0, sadness=1, joy=2, neutral=3

EMOTION2ID = {
    "anger": 0,
    "sadness": 1,
    "joy": 2,
    "neutral": 3,
}

# Map EmpatheticDialogues "context" strings → 4 emotion categories
# anger=0, sadness=1, joy=2, neutral=3
CONTEXT_TO_LABEL = {
    # anger-ish
    "angry": 0,
    "annoyed": 0,
    "furious": 0,
    "jealous": 0,
    "disgusted": 0,
    "embarrassed": 0,
    "impatient": 0,
    "afraid": 0,
    "terrified": 0,

    # sadness-ish
    "sad": 1,
    "disappointed": 1,
    "lonely": 1,
    "guilty": 1,
    "devastated": 1,
    "ashamed": 1,
    "nostalgic": 1,
    "apprehensive": 1,

    # joy-ish
    "happy": 2,
    "excited": 2,
    "hopeful": 2,
    "grateful": 2,
    "proud": 2,
    "content": 2,
    "confident": 2,
    "impressed": 2,
    "joyful": 2,
    "surprised": 2,

    # more "positive / warm but not very strong" → 先归到 neutral
    "caring": 3,
    "trusting": 3,
    "prepared": 3,
    "anticipating": 3,
    "sentimental": 3,
}

def map_context_to_label(context: str) -> int:
    c = context.strip().lower()
    return CONTEXT_TO_LABEL.get(c, 3)   # default = neutral


# -----------------------------
# 2. Load data (FIXED)
# -----------------------------
# CSV columns in your dataset:
# conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags

def load_csv_data(path: str) -> Tuple[List[str], List[int]]:
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            context = row["context"]
            utterance = row["utterance"]

            # Combine context + utterance as text input
            text = (context + " " + utterance).strip()

            # Convert context → label (0/1/2/3)
            label = map_context_to_label(context)

            texts.append(text)
            labels.append(label)

    return texts, labels


# -----------------------------
# 3. Build vocabulary
# -----------------------------

def build_vocab(texts: List[str], vocab_size: int = 10000) -> dict:
    counter = Counter()
    for t in texts:
        tokens = simple_tokenize(t)
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (tok, freq) in enumerate(counter.most_common(vocab_size - 2), start=2):
        vocab[tok] = i
    return vocab


# -----------------------------
# 4. Training loop
# -----------------------------

def train_model(
    data_csv_path: str,
    model_out_path: str = "models/emotion_lstm.pt",
    vocab_out_path: str = "models/vocab.pkl",
    batch_size: int = 64,
    lr: float = 1e-3,
    num_epochs: int = 10, #can be increased
    max_len: int = 50,
):
    # 1. Load data
    texts, labels = load_csv_data(data_csv_path)
    print(f"Loaded {len(texts)} examples.")

    # 2. Build vocab
    vocab = build_vocab(texts, vocab_size=10000)
    vocab_size = max(vocab.values()) + 1
    print(f"Vocab size: {vocab_size}")

    # 3. Create Dataset
    dataset = TextEmotionDataset(
        texts=texts,
        labels=labels,
        vocab=vocab,
        max_len=max_len,
        pad_idx=0,
        unk_idx=1,
    )

    # 4. Train/val split
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 5. Model, loss, optimizer
    model = LSTMEmotionClassifier(
        vocab_size=vocab_size,
        embed_dim=128, #can be increased
        hidden_dim=256,
        num_layers=1,
        num_classes=len(EMOTION2ID),
        pad_idx=0,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        avg_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # 7. Save model + vocab
    import os
    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), model_out_path)
    with open(vocab_out_path, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Saved model to {model_out_path}")
    print(f"Saved vocab to {vocab_out_path}")


if __name__ == "__main__":
    train_model("data/empathetic_train.csv")