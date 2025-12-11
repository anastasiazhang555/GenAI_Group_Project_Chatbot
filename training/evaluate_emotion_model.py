# training/evaluate_final_model.py

"""
Evaluate the *deployed* LSTM emotion classifier (emotion_lstm.pt)
on the existing test set, without retraining.

It reuses the same CSV loading logic as train_emotion_model.py
so that labels are consistent with training.
"""

from collections import Counter

from app.emotion_model import load_emotion_pipeline
from training.train_emotion_model import EMOTION2ID, load_csv_data


def evaluate_on_test(csv_path: str = "data/empathetic_test.csv"):
    # 1. Build inverse mapping: id -> label string
    id2label = {v: k for k, v in EMOTION2ID.items()}

    print(f"Loading test set from: {csv_path}")
    texts, label_ids = load_csv_data(csv_path)
    assert len(texts) == len(label_ids)
    print(f"Loaded {len(texts)} examples from test set.\n")

    # 2. Load the *final* deployed model + vocab
    emotion_pipeline = load_emotion_pipeline(
        checkpoint_path="models/emotion_lstm.pt",
        vocab_path="models/vocab.pkl",
        id2label=id2label,
        embed_dim=128,   # must match training
        hidden_dim=256,  # must match training
    )

    # 3. Iterate over test set and collect metrics
    total = len(texts)
    correct = 0

    per_class_total = Counter()
    per_class_correct = Counter()

    for text, true_id in zip(texts, label_ids):
        true_label = id2label[true_id]

        pred_label, conf = emotion_pipeline.predict(text)

        per_class_total[true_label] += 1
        if pred_label == true_label:
            correct += 1
            per_class_correct[true_label] += 1

    # 4. Print results
    overall_acc = correct / total if total > 0 else 0.0
    print("=== Final Deployed Model Evaluation ===")
    print(f"Overall accuracy: {overall_acc:.4f} ({correct}/{total})\n")

    print("Per-class accuracy:")
    for lbl in sorted(per_class_total.keys()):
        t = per_class_total[lbl]
        c = per_class_correct[lbl]
        acc = c / t if t > 0 else 0.0
        print(f"  {lbl:8s}: {acc:.4f} ({c}/{t})")


if __name__ == "__main__":
    #   data/empathetic_test.csv
    evaluate_on_test("data/empathetic_test.csv")