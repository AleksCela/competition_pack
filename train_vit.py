import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score
from PIL import Image


MODEL_NAME = "google/vit-base-patch16-224"
MAX_IMAGES = 10_000
SEED = 42


def find_image_column(ds):
    for col, feature in ds.features.items():
        if feature.__class__.__name__ == "Image":
            return col
    raise ValueError(f"No image column found. Available columns: {ds.column_names}")


def is_rgb(example, image_col):
    try:
        image_obj = example[image_col]
        if not isinstance(image_obj, Image.Image):
            return False
        return image_obj.mode == "RGB"
    except Exception:
        return False


def transform(batch, processor, image_col):
    # Keep only proper RGB tensors for the model
    rgb_images = [img for img in batch[image_col]]
    inputs = processor(rgb_images, return_tensors="pt")
    batch["pixel_values"] = inputs["pixel_values"]
    return batch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    logits, labels_true = eval_pred
    preds = logits.argmax(axis=1)
    return {"accuracy": accuracy_score(labels_true, preds)}


def get_backbone_module(model):
    if hasattr(model, "vit"):
        return model.vit
    if hasattr(model, "base_model"):
        return model.base_model
    raise ValueError("Could not find ViT backbone module to freeze/unfreeze.")


def set_backbone_trainable(model, trainable):
    backbone = get_backbone_module(model)
    for param in backbone.parameters():
        param.requires_grad = trainable


parser = argparse.ArgumentParser(description="Train ViT on PlantVillage")
parser.add_argument(
    "--max-images",
    type=int,
    default=MAX_IMAGES,
    help="Cap number of RGB images used. Use 0 or negative for full dataset.",
)
args = parser.parse_args()

# 1) Load full PlantVillage train data
dataset = load_dataset("imagefolder", data_dir="data/train")
raw_train = dataset["train"]
image_column = find_image_column(raw_train)
labels = raw_train.features["label"].names

# 2) Keep only RGB samples, then limit to 10k for fast hackathon iteration
rgb_only = raw_train.filter(lambda x: is_rgb(x, image_column))
rgb_only = rgb_only.shuffle(seed=SEED)
if args.max_images and args.max_images > 0:
    limit = min(args.max_images, len(rgb_only))
    rgb_only = rgb_only.select(range(limit))
else:
    limit = len(rgb_only)

# 3) Train/val split from that 10k set
splits = rgb_only.train_test_split(test_size=0.1, seed=SEED, stratify_by_column="label")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
train_dataset = splits["train"].with_transform(lambda b: transform(b, processor, image_column))
eval_dataset = splits["test"].with_transform(lambda b: transform(b, processor, image_column))


def build_model():
    return AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )


model = build_model()

# Baseline eval before any training (head is random for 38 classes)
baseline_args = TrainingArguments(
    output_dir="./vit-model/baseline",
    per_device_eval_batch_size=16,
    remove_unused_columns=False,
    report_to="none",
)
baseline_trainer = Trainer(
    model=model,
    args=baseline_args,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)
baseline_metrics = baseline_trainer.evaluate()
print("Baseline (no fine-tune) metrics:", baseline_metrics)

# Phase 1: train classifier only (backbone frozen)
set_backbone_trainable(model, trainable=False)
phase1_args = TrainingArguments(
    output_dir="./vit-model/phase1_linear_probe",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    learning_rate=1e-3,
    weight_decay=0.01,
    warmup_ratio=0.05,
    fp16=torch.cuda.is_available(),
    logging_steps=25,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
)
phase1_trainer = Trainer(
    model=model,
    args=phase1_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)
phase1_trainer.train()
phase1_metrics = phase1_trainer.evaluate()
print("Phase 1 (frozen backbone) metrics:", phase1_metrics)
phase1_trainer.save_model("./vit-model/phase1_linear_probe")
processor.save_pretrained("./vit-model/phase1_linear_probe")

# Phase 2: full fine-tuning
set_backbone_trainable(model, trainable=True)
phase2_args = TrainingArguments(
    output_dir="./vit-model/phase2_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    logging_steps=25,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
)
phase2_trainer = Trainer(
    model=model,
    args=phase2_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)
phase2_trainer.train()
phase2_metrics = phase2_trainer.evaluate()
print("Phase 2 (full fine-tune) metrics:", phase2_metrics)
phase2_trainer.save_model("./vit-model/phase2_finetuned")
processor.save_pretrained("./vit-model/phase2_finetuned")

print(f"Done. RGB-only samples used: {limit}")