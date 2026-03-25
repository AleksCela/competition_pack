import argparse
import csv
import random
import shutil
from pathlib import Path


def collect_images(train_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = []
    for class_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for img_path in class_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in exts:
                samples.append((img_path, label))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Create a mock unlabeled test set from training images.")
    parser.add_argument("--train-dir", default="data/train", help="Labeled training directory")
    parser.add_argument("--out-dir", default="data/test_mock", help="Output mock test directory")
    parser.add_argument("--labels-csv", default="data/test_mock_labels.csv", help="Ground truth CSV for local scoring")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    labels_csv = Path(args.labels_csv)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    all_samples = collect_images(train_dir)
    if not all_samples:
        raise ValueError(f"No images found in {train_dir}")

    random.seed(args.seed)
    k = min(args.num_images, len(all_samples))
    chosen = random.sample(all_samples, k=k)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (src, label) in enumerate(chosen, start=1):
        new_name = f"img_{idx:05d}{src.suffix.lower()}"
        dst = out_dir / new_name
        shutil.copy2(src, dst)
        rows.append({"image_id": new_name, "label": label})

    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created mock test set: {out_dir} ({k} images)")
    print(f"Ground-truth labels saved to: {labels_csv}")


if __name__ == "__main__":
    main()
