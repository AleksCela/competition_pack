import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def list_image_files(test_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_rgb(path: Path):
    with Image.open(path) as img:
        return img.convert("RGB")


def find_test_dir(user_arg: str):
    requested = Path(user_arg)
    if requested.exists() and requested.is_dir():
        return requested

    candidates = [
        Path("data/test"),
        Path("test"),
        Path("dataset/test"),
        Path("competition/test"),
        Path("data/public_test"),
        Path("public_test"),
    ]

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            print(f"[info] --test-dir not found, using discovered folder: {cand}")
            return cand

    return requested


def main():
    parser = argparse.ArgumentParser(description="Predict labels for test images and write submission CSV.")
    parser.add_argument("--model-dir", default="vit-model/phase2_finetuned", help="Path to trained model directory")
    parser.add_argument("--test-dir", default="data/test", help="Directory with unlabeled test images")
    parser.add_argument("--output-csv", default="submission.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument(
        "--processor-name",
        default="google/vit-base-patch16-224",
        help="Fallback image processor if model-dir has no preprocessor_config.json",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    test_dir = find_test_dir(args.test_dir)
    output_csv = Path(args.output_csv)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            "Put competition test images in a folder and pass it via --test-dir, e.g.\n"
            "python predict_test.py --model-dir vit-model/phase2_finetuned --test-dir <path_to_test_images>"
        )

    image_paths = list_image_files(test_dir)
    if not image_paths:
        raise ValueError(f"No images found in {test_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = AutoImageProcessor.from_pretrained(model_dir)
    except OSError:
        print(
            f"[info] No processor config in {model_dir}. Falling back to: {args.processor_name}"
        )
        processor = AutoImageProcessor.from_pretrained(args.processor_name)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    id2label = model.config.id2label

    rows = []
    with torch.no_grad():
        for batch_paths in chunked(image_paths, args.batch_size):
            images = [load_rgb(p) for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits
            pred_ids = logits.argmax(dim=-1).tolist()

            for path, pred_id in zip(batch_paths, pred_ids):
                label = id2label[int(pred_id)]
                rows.append({"image_id": path.name, "label": label})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} predictions to {output_csv}")


if __name__ == "__main__":
    main()
