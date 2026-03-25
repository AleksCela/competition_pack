"""Analyze model confusion — find exactly where your model is weak.

Run this after training to see which class pairs get confused.
Then retrain with --weighted-loss or focus your prompt engineering there.
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image


def find_image_column(ds):
    for col, feature in ds.features.items():
        if feature.__class__.__name__ == "Image":
            return col
    raise ValueError(f"No image column found")


def is_rgb(example, image_col):
    try:
        img = example[image_col]
        return isinstance(img, Image.Image) and img.mode == "RGB"
    except Exception:
        return False


def transform(batch, processor, image_col):
    inputs = processor(batch[image_col], return_tensors="pt")
    batch["pixel_values"] = inputs["pixel_values"]
    return batch


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Confusion matrix analysis")
    parser.add_argument("--model-dir", required=True, help="Trained model directory")
    parser.add_argument("--data-dir", default="data/train", help="Training data (will split off val)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=0, help="Cap dataset (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = AutoModelForImageClassification.from_pretrained(args.model_dir)
    model.to(device).eval()
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_dir)
    except Exception:
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    id2label = model.config.id2label
    labels_list = [id2label[i] for i in range(len(id2label))]

    # Load validation data
    dataset = load_dataset("imagefolder", data_dir=args.data_dir)
    raw = dataset["train"]
    image_col = find_image_column(raw)
    rgb_only = raw.filter(lambda x: is_rgb(x, image_col)).shuffle(seed=args.seed)
    if args.max_images > 0:
        rgb_only = rgb_only.select(range(min(args.max_images, len(rgb_only))))

    splits = rgb_only.train_test_split(test_size=0.1, seed=args.seed, stratify_by_column="label")
    eval_ds = splits["test"].with_transform(lambda b: transform(b, processor, image_col))

    print(f"Evaluating on {len(splits['test'])} validation images...")

    # Run predictions
    all_preds = []
    all_labels = []
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(eval_ds, batch_size=args.batch_size, collate_fn=collate_fn)
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            logits = model(pixel_values=pv).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n{'='*80}")
    print("PER-CLASS ACCURACY (sorted worst → best):")
    print(f"{'='*80}")
    class_acc = []
    for i in range(len(labels_list)):
        total = cm[i].sum()
        correct = cm[i][i]
        ca = correct / max(total, 1)
        class_acc.append((ca, total, labels_list[i]))

    class_acc.sort()
    for ca, total, name in class_acc:
        bar = "#" * int(ca * 40)
        flag = " ← WEAK" if ca < 0.85 else ""
        print(f"  {name:50s} {ca:5.1%} ({total:4d} samples) {bar}{flag}")

    # Most confused pairs
    print(f"\n{'='*80}")
    print("MOST CONFUSED PAIRS:")
    print(f"{'='*80}")
    confused = []
    for i in range(len(labels_list)):
        for j in range(len(labels_list)):
            if i != j and cm[i][j] > 0:
                confused.append((cm[i][j], labels_list[i], labels_list[j]))
    confused.sort(reverse=True)

    for count, true_cls, pred_cls in confused[:20]:
        print(f"  {true_cls:45s} → {pred_cls:45s}  ({count} errors)")

    # Recommendations
    weak_classes = [name for ca, total, name in class_acc if ca < 0.85]
    if weak_classes:
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS:")
        print(f"{'='*80}")
        print(f"  {len(weak_classes)} classes below 85% accuracy:")
        for name in weak_classes:
            print(f"    - {name}")
        print(f"\n  Actions to try:")
        print(f"    1. Retrain with --weighted-loss to boost weak classes")
        print(f"    2. Add more augmentation for these specific classes")
        print(f"    3. If ensembling, check if your second model is better on these")
    else:
        print(f"\nAll classes above 85% — looking good!")

    # Save report
    report = {
        "overall_accuracy": acc,
        "per_class": [{"class": n, "accuracy": a, "samples": int(t)} for a, t, n in sorted(class_acc)],
        "top_confused": [{"true": t, "predicted": p, "count": int(c)} for c, t, p in confused[:30]],
        "weak_classes": weak_classes,
    }
    out_path = Path(args.model_dir) / "confusion_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
