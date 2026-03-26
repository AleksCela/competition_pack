"""CLIP zero-shot classification — first submission, no training needed.

Gets 60-75% on PlantVillage. Takes ~5 minutes on GPU, ~20 on CPU.
Run this FIRST to get on the leaderboard while your fine-tune trains.
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# PlantVillage class names — used as zero-shot targets
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "other",
]

# Custom prompts for the "other" class — not a plant leaf at all
OTHER_PROMPTS = [
    "a photo of something that is not a plant",
    "a random image unrelated to plants or leaves",
    "a photo of a road, pothole, or urban scene",
    "an image that is not a crop leaf or plant disease",
    "a non-plant photograph",
]


def build_prompts(class_names):
    """Multiple prompt templates — averaged for better accuracy."""
    templates = [
        "a photo of a {} leaf",
        "a close-up photo of {} on a plant leaf",
        "{} disease on a plant leaf",
        "a botanical photo showing {}",
        "a leaf affected by {}",
    ]
    prompts_per_class = {}
    for cls in class_names:
        if cls == "other":
            prompts_per_class[cls] = OTHER_PROMPTS
            continue

        clean = cls.replace("___", " ").replace("_", " ").replace("  ", " ").strip()
        parts = clean.split(" ", 1)
        plant = parts[0]
        condition = parts[1] if len(parts) > 1 else "healthy"

        class_prompts = []
        for tmpl in templates:
            class_prompts.append(tmpl.format(clean))
        # Add plant-specific prompts
        class_prompts.append(f"a {plant} leaf with {condition}")
        class_prompts.append(f"{condition} on {plant}")
        prompts_per_class[cls] = class_prompts
    return prompts_per_class


def list_image_files(test_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main():
    parser = argparse.ArgumentParser(description="CLIP zero-shot plant disease classification")
    parser.add_argument("--test-dir", default="data/test", help="Directory with test images")
    parser.add_argument("--output-csv", default="submission_clip.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--model-name", default="openai/clip-vit-large-patch14", help="CLIP model to use")
    parser.add_argument(
        "--classes-from-model", default=None,
        help="Path to a trained model dir to read class names from config instead of hardcoded list",
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    image_paths = list_image_files(test_dir)
    if not image_paths:
        raise ValueError(f"No images found in {test_dir}")

    # Determine class names
    class_names = PLANTVILLAGE_CLASSES
    if args.classes_from_model:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.classes_from_model)
        if hasattr(config, "id2label") and config.id2label:
            class_names = [config.id2label[i] for i in range(len(config.id2label))]
            print(f"[info] Loaded {len(class_names)} class names from {args.classes_from_model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Classes: {len(class_names)}")
    print(f"Test images: {len(image_paths)}")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    # Pre-compute text embeddings with prompt ensemble
    print("Computing text embeddings with prompt ensemble...")
    prompts_per_class = build_prompts(class_names)
    text_features_avg = []
    with torch.no_grad():
        for cls in class_names:
            prompts = prompts_per_class[cls]
            inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            avg_feat = feats.mean(dim=0)
            avg_feat = avg_feat / avg_feat.norm()
            text_features_avg.append(avg_feat)
    text_features = torch.stack(text_features_avg)  # (38, dim)

    # Classify images
    rows = []
    done = 0
    with torch.no_grad():
        for batch_paths in chunked(image_paths, args.batch_size):
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarities = image_features @ text_features.T  # (batch, 38)
            pred_ids = similarities.argmax(dim=-1).tolist()

            for path, pred_id in zip(batch_paths, pred_ids):
                rows.append({"id": path.name, "label": class_names[pred_id]})

            done += len(batch_paths)
            if done % 100 == 0 or done == len(image_paths):
                print(f"  {done}/{len(image_paths)}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Wrote {len(rows)} predictions to {output_csv}")
    print("Upload this NOW while you train your fine-tuned model.")


if __name__ == "__main__":
    main()
