"""Predict with Test-Time Augmentation (TTA).

Runs each image through multiple augmented views and averages the probabilities.
Typically gains +1-2% accuracy for free. Use this instead of plain predict_test.py.

Supports both standard HF models (ViT, Swin) and custom DINOv2Classifier.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torchvision.transforms as T


# TTA augmentations — each produces a different view of the same image
TTA_TRANSFORMS = [
    T.Compose([T.Resize(256), T.CenterCrop(224)]),                          # original
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0)]),  # h-flip
    T.Compose([T.Resize(256), T.FiveCrop(224), lambda crops: crops[0]]),    # top-left
    T.Compose([T.Resize(256), T.FiveCrop(224), lambda crops: crops[1]]),    # top-right
    T.Compose([T.Resize(256), T.FiveCrop(224), lambda crops: crops[4]]),    # center
    T.Compose([T.Resize(288), T.CenterCrop(224)]),                          # slight zoom
]


def list_image_files(test_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        [p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name,
    )


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_model(model_dir, model_type, device):
    """Load model — supports HF models and custom DINOv2Classifier."""
    model_dir = Path(model_dir)

    if model_type == "dinov2":
        from train_dinov2 import DINOv2Classifier
        model = DINOv2Classifier.load_pretrained(model_dir)
        model.to(device).eval()
        # Read id2label from config
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label", {})
        # Try loading processor from model dir, fall back to backbone name
        try:
            processor = AutoImageProcessor.from_pretrained(model_dir)
        except Exception:
            processor = AutoImageProcessor.from_pretrained(cfg.get("backbone", "facebook/dinov2-large"))
        return model, processor, id2label

    # Standard HF model
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.to(device).eval()
    try:
        processor = AutoImageProcessor.from_pretrained(model_dir)
    except Exception:
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    id2label = model.config.id2label
    return model, processor, id2label


def main():
    parser = argparse.ArgumentParser(description="Predict with TTA")
    parser.add_argument("--model-dir", required=True, help="Trained model directory")
    parser.add_argument("--test-dir", default="data/test", help="Test images directory")
    parser.add_argument("--output-csv", default="submission_tta.csv", help="Output CSV")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--model-type", default="auto", choices=["auto", "hf", "dinov2"],
                        help="Model type (auto-detects from config.json)")
    parser.add_argument("--num-tta", type=int, default=6, help="Number of TTA views (1-6, 1=no TTA)")
    parser.add_argument(
        "--ood-threshold", type=float, default=0.0,
        help="If max softmax confidence < this value, predict 'other' instead. "
             "0.0 = disabled. Try 0.5 as a starting point.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    test_dir = Path(args.test_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    # Auto-detect model type
    model_type = args.model_type
    if model_type == "auto":
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            model_type = "dinov2" if cfg.get("model_type") == "dinov2_classifier" else "hf"
        else:
            model_type = "hf"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    print(f"TTA views: {args.num_tta}")

    model, processor, id2label = load_model(model_dir, model_type, device)
    image_paths = list_image_files(test_dir)
    print(f"Test images: {len(image_paths)}")

    # Find the index for "other" in the label map (needed for OOD fallback)
    other_label = "other"
    label2id = {v: k for k, v in (id2label.items() if isinstance(list(id2label.keys())[0], int) else {int(k): v for k, v in id2label.items()}.items())}
    ood_threshold = args.ood_threshold
    if ood_threshold > 0:
        if other_label not in {v for v in id2label.values()}:
            print(f"WARNING: 'other' class not found in model labels — OOD threshold disabled.")
            ood_threshold = 0.0
        else:
            print(f"OOD threshold: {ood_threshold:.2f} (low-confidence predictions → 'other')")

    tta_augs = TTA_TRANSFORMS[:args.num_tta]

    rows = []
    done = 0
    with torch.no_grad():
        for batch_paths in chunked(image_paths, args.batch_size):
            images_raw = [Image.open(p).convert("RGB") for p in batch_paths]

            # Accumulate probabilities across TTA views
            avg_probs = None
            for aug in tta_augs:
                augmented = [aug(img) for img in images_raw]
                inputs = processor(images=augmented, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                if model_type == "dinov2":
                    out = model(inputs["pixel_values"])
                else:
                    out = model(**inputs)

                probs = F.softmax(out.logits, dim=-1)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs = avg_probs + probs

            avg_probs = avg_probs / len(tta_augs)
            max_confs, pred_ids = avg_probs.max(dim=-1)
            max_confs = max_confs.tolist()
            pred_ids = pred_ids.tolist()

            for path, pred_id, confidence in zip(batch_paths, pred_ids, max_confs):
                if ood_threshold > 0 and confidence < ood_threshold:
                    label = other_label
                else:
                    label = id2label[str(pred_id)] if str(pred_id) in id2label else id2label.get(pred_id, str(pred_id))
                rows.append({"id": path.name, "label": label})

            done += len(batch_paths)
            if done % 100 == 0 or done == len(image_paths):
                print(f"  {done}/{len(image_paths)}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} predictions → {output_csv}")


if __name__ == "__main__":
    main()
