"""Ensemble predictions from multiple models.

Averages softmax probabilities across models, then takes argmax.
Different architectures make different mistakes — ensembling cancels them out.
Typical gain: +1-3% over single best model.

Usage:
    python ensemble_predict.py \
        --model-dirs vit-model/best dinov2-model/best \
        --model-types hf dinov2 \
        --test-dir data/test \
        --output-csv submission_ensemble.csv
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


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
    model_dir = Path(model_dir)

    if model_type == "dinov2":
        from train_dinov2 import DINOv2Classifier
        model = DINOv2Classifier.load_pretrained(model_dir)
        model.to(device).eval()
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label", {})
        try:
            processor = AutoImageProcessor.from_pretrained(model_dir)
        except Exception:
            processor = AutoImageProcessor.from_pretrained(cfg.get("backbone", "facebook/dinov2-large"))
        return model, processor, id2label, "dinov2"

    # Standard HF model
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.to(device).eval()
    try:
        processor = AutoImageProcessor.from_pretrained(model_dir)
    except Exception:
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    id2label = model.config.id2label
    return model, processor, id2label, "hf"


def detect_model_type(model_dir):
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "dinov2_classifier":
            return "dinov2"
    return "hf"


def main():
    parser = argparse.ArgumentParser(description="Ensemble multiple models")
    parser.add_argument("--model-dirs", nargs="+", required=True, help="Paths to model directories")
    parser.add_argument("--model-types", nargs="+", default=None,
                        help="Model types per dir (hf or dinov2). Auto-detected if omitted.")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Weight per model (default: equal). E.g. --weights 0.6 0.4")
    parser.add_argument("--test-dir", default="data/test", help="Test images directory")
    parser.add_argument("--output-csv", default="submission_ensemble.csv", help="Output CSV")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (lower for multiple models in memory)")
    parser.add_argument(
        "--ood-threshold", type=float, default=0.0,
        help="If ensemble max confidence < this, predict 'other'. 0.0 = disabled. Try 0.5.",
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    image_paths = list_image_files(test_dir)
    if not image_paths:
        raise ValueError(f"No images in {test_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_models = len(args.model_dirs)

    # Detect model types
    model_types = args.model_types or [detect_model_type(d) for d in args.model_dirs]
    if len(model_types) != n_models:
        raise ValueError(f"Got {len(model_types)} model types for {n_models} model dirs")

    # Set weights
    weights = args.weights or [1.0 / n_models] * n_models
    if len(weights) != n_models:
        raise ValueError(f"Got {len(weights)} weights for {n_models} models")
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    print(f"Device: {device}")
    print(f"Models: {n_models}")
    for i, (d, t, w) in enumerate(zip(args.model_dirs, model_types, weights)):
        print(f"  [{i+1}] {d} (type={t}, weight={w:.2f})")
    print(f"Test images: {len(image_paths)}")

    # Load all models
    models = []
    for model_dir, mtype in zip(args.model_dirs, model_types):
        m, p, id2label, actual_type = load_model(model_dir, mtype, device)
        models.append((m, p, id2label, actual_type))

    # Use id2label from first model
    master_id2label = models[0][2]
    other_label = "other"
    ood_threshold = args.ood_threshold
    if ood_threshold > 0:
        if other_label not in {v for v in master_id2label.values()}:
            print("WARNING: 'other' not in model labels — OOD threshold disabled.")
            ood_threshold = 0.0
        else:
            print(f"OOD threshold: {ood_threshold:.2f}")

    rows = []
    done = 0
    with torch.no_grad():
        for batch_paths in chunked(image_paths, args.batch_size):
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            ensemble_probs = None

            for (model, processor, id2label, mtype), weight in zip(models, weights):
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                if mtype == "dinov2":
                    out = model(inputs["pixel_values"])
                else:
                    out = model(**inputs)

                probs = F.softmax(out.logits, dim=-1) * weight
                if ensemble_probs is None:
                    ensemble_probs = probs
                else:
                    ensemble_probs = ensemble_probs + probs

            max_confs, pred_ids = ensemble_probs.max(dim=-1)
            for path, pid, confidence in zip(batch_paths, pred_ids.tolist(), max_confs.tolist()):
                if ood_threshold > 0 and confidence < ood_threshold:
                    label = other_label
                else:
                    label = master_id2label[str(pid)] if str(pid) in master_id2label else master_id2label.get(pid, str(pid))
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
