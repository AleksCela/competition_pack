"""Fine-tune ViT-large (or any HF vision model) on PlantVillage.

Improvements over the basic train_vit.py:
- Supports vit-large, dinov2, swin, or any AutoModelForImageClassification
- Weighted loss for class imbalance
- Gradient accumulation for larger effective batch sizes on small GPUs
- Data augmentation (random crop, flip, color jitter)
- Saves confusion matrix after training
- Single best model saved to vit-model/best/
"""

import argparse
import json
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import torchvision.transforms as T


def find_image_column(ds):
    for col, feature in ds.features.items():
        if feature.__class__.__name__ == "Image":
            return col
    raise ValueError(f"No image column found. Columns: {ds.column_names}")


def is_rgb(example, image_col):
    try:
        img = example[image_col]
        return isinstance(img, Image.Image) and img.mode == "RGB"
    except Exception:
        return False


def make_train_augment(crop_size):
    return T.Compose([
        T.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    ])


def get_crop_size(processor):
    """Read the expected input resolution from the processor config."""
    for attr in ("size", "crop_size"):
        val = getattr(processor, attr, None)
        if val is None:
            continue
        if isinstance(val, dict):
            return val.get("shortest_edge") or val.get("height") or val.get("width") or 224
        if isinstance(val, int):
            return val
    return 224


def transform_train(batch, processor, image_col, crop_size):
    augment = make_train_augment(crop_size)
    images = [augment(img) for img in batch[image_col]]
    inputs = processor(images, return_tensors="pt")
    batch["pixel_values"] = inputs["pixel_values"]
    return batch


def transform_eval(batch, processor, image_col):
    inputs = processor(batch[image_col], return_tensors="pt")
    batch["pixel_values"] = inputs["pixel_values"]
    return batch


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


class WeightedLossTrainer(Trainer):
    """Trainer with per-class weighted cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def get_backbone(model):
    for attr in ["vit", "dinov2", "swinv2", "swin", "vision_model", "siglip_vision_model", "base_model"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    return None


def set_backbone_trainable(model, trainable):
    backbone = get_backbone(model)
    if backbone:
        for p in backbone.parameters():
            p.requires_grad = trainable


def main():
    parser = argparse.ArgumentParser(description="Fine-tune vision model on PlantVillage")
    parser.add_argument("--model", default="google/vit-large-patch16-224", help="HF model name or path")
    parser.add_argument("--data-dir", default="data/train", help="Training data in ImageFolder format")
    parser.add_argument("--output-dir", default="vit-model", help="Output directory root")
    parser.add_argument("--max-images", type=int, default=0, help="Cap dataset size (0=all)")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Per-device eval batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--weighted-loss", action="store_true", help="Use class-weighted loss")
    parser.add_argument("--no-phase1", action="store_true", help="Skip linear probe phase, go straight to full fine-tune")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_name}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load data
    print(f"Loading dataset from {args.data_dir}...")
    dataset = load_dataset("imagefolder", data_dir=args.data_dir)
    raw_train = dataset["train"]
    image_col = find_image_column(raw_train)
    labels = raw_train.features["label"].names
    num_classes = len(labels)
    print(f"Classes: {num_classes}")
    print(f"Total images: {len(raw_train)}")

    # Filter RGB
    rgb_only = raw_train.filter(lambda x: is_rgb(x, image_col))
    rgb_only = rgb_only.shuffle(seed=args.seed)
    if args.max_images and args.max_images > 0:
        rgb_only = rgb_only.select(range(min(args.max_images, len(rgb_only))))
    print(f"RGB images used: {len(rgb_only)}")

    # Split
    splits = rgb_only.train_test_split(test_size=0.1, seed=args.seed, stratify_by_column="label")

    # Compute class weights
    class_weights = None
    if args.weighted_loss:
        label_counts = Counter(splits["train"]["label"])
        total = sum(label_counts.values())
        class_weights = [total / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)]
        print(f"Using weighted loss. Max weight: {max(class_weights):.2f}, Min: {min(class_weights):.2f}")

    # Processor + transforms
    processor = AutoImageProcessor.from_pretrained(args.model)
    crop_size = get_crop_size(processor)
    print(f"Input crop size: {crop_size}px")
    train_ds = splits["train"].with_transform(lambda b: transform_train(b, processor, image_col, crop_size))
    eval_ds = splits["test"].with_transform(lambda b: transform_eval(b, processor, image_col))

    # Build model
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=num_classes,
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )

    best_dir = f"{args.output_dir}/best"

    # Phase 1: linear probe (optional, skip with --no-phase1)
    if not args.no_phase1:
        print("\n--- Phase 1: Linear Probe (backbone frozen) ---")
        set_backbone_trainable(model, False)
        p1_args = TrainingArguments(
            output_dir=f"{args.output_dir}/phase1",
            per_device_train_batch_size=args.batch_size * 2,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=2,
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_ratio=0.05,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            seed=args.seed,
        )
        trainer1 = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=p1_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
        )
        trainer1.train()
        metrics1 = trainer1.evaluate()
        print(f"Phase 1 accuracy: {metrics1['eval_accuracy']:.4f}")

    # Phase 2: full fine-tune
    print("\n--- Phase 2: Full Fine-Tune ---")
    set_backbone_trainable(model, True)
    p2_args = TrainingArguments(
        output_dir=f"{args.output_dir}/phase2",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    trainer2 = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=p2_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    trainer2.train()
    metrics2 = trainer2.evaluate()
    print(f"Phase 2 accuracy: {metrics2['eval_accuracy']:.4f}")

    # Save best model
    trainer2.save_model(best_dir)
    processor.save_pretrained(best_dir)
    print(f"\nBest model saved to {best_dir}/")

    # Save confusion matrix
    print("\nGenerating confusion matrix on validation set...")
    preds_output = trainer2.predict(eval_ds)
    preds = preds_output.predictions.argmax(axis=1)
    true_labels = preds_output.label_ids
    cm = confusion_matrix(true_labels, preds)

    # Find most confused pairs
    confused = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i][j] > 0:
                confused.append((cm[i][j], labels[i], labels[j]))
    confused.sort(reverse=True)

    print("\nTop 10 most confused class pairs:")
    for count, true_cls, pred_cls in confused[:10]:
        print(f"  {true_cls:45s} → {pred_cls:45s}  ({count} errors)")

    # Save confusion info to file
    confusion_file = f"{args.output_dir}/confusion_report.json"
    report = {
        "val_accuracy": metrics2["eval_accuracy"],
        "num_classes": num_classes,
        "top_confused_pairs": [
            {"true": t, "predicted": p, "count": int(c)} for c, t, p in confused[:20]
        ],
    }
    with open(confusion_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Confusion report saved to {confusion_file}")

    print(f"\nDone! Final val accuracy: {metrics2['eval_accuracy']:.4f}")
    print(f"Model ready at: {best_dir}/")
    print(f"Next step: python predict_tta.py --model-dir {best_dir} --test-dir data/test")


if __name__ == "__main__":
    main()
