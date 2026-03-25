"""Fine-tune DINOv2-large for PlantVillage classification.

DINOv2 has excellent visual features but no built-in classification head.
This script adds a linear head and fine-tunes the whole model.
Often beats ViT on fine-grained tasks like plant disease.
"""

import argparse
import json
import torch
import torch.nn as nn
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import torchvision.transforms as T


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + linear classification head."""

    def __init__(self, model_name, num_labels, id2label=None, label2id=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        # Store config-like attributes for compatibility
        self.config = self.backbone.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label or {}
        self.config.label2id = label2id or {}

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values=pixel_values)
        # Use CLS token
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return type("Output", (), {"loss": loss, "logits": logits})()

    def save_pretrained(self, path):
        import os, json as _json
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        # Save config for predict_tta.py compatibility
        cfg = {
            "model_type": "dinov2_classifier",
            "backbone": self.backbone.config._name_or_path,
            "num_labels": self.num_labels,
            "id2label": self.config.id2label,
            "label2id": self.config.label2id,
            "hidden_size": self.backbone.config.hidden_size,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            _json.dump(cfg, f, indent=2)

    @classmethod
    def load_pretrained(cls, path):
        import os, json as _json
        with open(os.path.join(path, "config.json")) as f:
            cfg = _json.load(f)
        model = cls(
            cfg["backbone"], cfg["num_labels"],
            id2label=cfg.get("id2label"), label2id=cfg.get("label2id"),
        )
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state)
        return model


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


TRAIN_AUGMENT = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
])


def transform_train(batch, processor, image_col):
    images = [TRAIN_AUGMENT(img) for img in batch[image_col]]
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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 on PlantVillage")
    parser.add_argument("--model", default="facebook/dinov2-large", help="DINOv2 model name")
    parser.add_argument("--data-dir", default="data/train", help="Training data dir")
    parser.add_argument("--output-dir", default="dinov2-model", help="Output directory")
    parser.add_argument("--max-images", type=int, default=0, help="Cap dataset (0=all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Train batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    dataset = load_dataset("imagefolder", data_dir=args.data_dir)
    raw = dataset["train"]
    image_col = find_image_column(raw)
    labels = raw.features["label"].names
    num_classes = len(labels)
    print(f"Classes: {num_classes}, Total: {len(raw)}")

    rgb_only = raw.filter(lambda x: is_rgb(x, image_col)).shuffle(seed=args.seed)
    if args.max_images > 0:
        rgb_only = rgb_only.select(range(min(args.max_images, len(rgb_only))))
    print(f"Using {len(rgb_only)} images")

    splits = rgb_only.train_test_split(test_size=0.1, seed=args.seed, stratify_by_column="label")

    processor = AutoImageProcessor.from_pretrained(args.model)
    train_ds = splits["train"].with_transform(lambda b: transform_train(b, processor, image_col))
    eval_ds = splits["test"].with_transform(lambda b: transform_eval(b, processor, image_col))

    # Build model
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}
    model = DINOv2Classifier(args.model, num_classes, id2label=id2label, label2id=label2id)

    best_dir = f"{args.output_dir}/best"

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/checkpoints",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal accuracy: {metrics['eval_accuracy']:.4f}")

    # Save
    model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)
    print(f"Model saved to {best_dir}/")

    # Confusion report
    preds_out = trainer.predict(eval_ds)
    preds = preds_out.predictions.argmax(axis=1)
    cm = confusion_matrix(preds_out.label_ids, preds)
    confused = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i][j] > 0:
                confused.append((cm[i][j], labels[i], labels[j]))
    confused.sort(reverse=True)
    print("\nTop confused pairs:")
    for c, t, p in confused[:10]:
        print(f"  {t:45s} → {p:45s}  ({c})")

    print(f"\nNext: python predict_tta.py --model-dir {best_dir} --test-dir data/test --model-type dinov2")


if __name__ == "__main__":
    main()
