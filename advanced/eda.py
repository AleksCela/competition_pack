"""Quick EDA — run this FIRST on competition day to understand the data."""

import argparse
from pathlib import Path
from collections import Counter
from PIL import Image


def scan_images(data_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    files = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files, key=lambda p: p.name)


def main():
    parser = argparse.ArgumentParser(description="Quick dataset EDA")
    parser.add_argument("--data-dir", required=True, help="Path to data folder (train or test)")
    parser.add_argument("--sample", type=int, default=20, help="Number of images to inspect closely")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    images = scan_images(data_dir)
    print(f"\n{'='*60}")
    print(f"Dataset: {data_dir}")
    print(f"Total images found: {len(images)}")
    print(f"{'='*60}")

    # Check if it's ImageFolder format (subdirectories = classes)
    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if subdirs:
        print(f"\nSubdirectories (potential classes): {len(subdirs)}")
        class_counts = {}
        for d in subdirs:
            count = len(scan_images(d))
            class_counts[d.name] = count
        print(f"\nClass distribution:")
        for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            bar = "#" * min(count // 20, 50)
            print(f"  {name:45s} {count:5d}  {bar}")
        counts = list(class_counts.values())
        print(f"\nMin class size: {min(counts)}")
        print(f"Max class size: {max(counts)}")
        print(f"Imbalance ratio: {max(counts)/max(min(counts),1):.1f}x")
    else:
        print("\nFlat directory (no subdirectories) — likely unlabeled test set")

    # Inspect sample images
    print(f"\nInspecting first {min(args.sample, len(images))} images:")
    sizes = []
    modes = Counter()
    formats = Counter()
    for img_path in images[:args.sample]:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
                modes[img.mode] += 1
                formats[img.format or "unknown"] += 1
        except Exception as e:
            print(f"  ERROR reading {img_path.name}: {e}")

    if sizes:
        unique_sizes = set(sizes)
        print(f"  Image modes: {dict(modes)}")
        print(f"  Image formats: {dict(formats)}")
        if len(unique_sizes) == 1:
            print(f"  All same size: {sizes[0]}")
        else:
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            print(f"  Size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
            print(f"  Unique sizes: {len(unique_sizes)}")

    # Check file extensions
    ext_counts = Counter(p.suffix.lower() for p in images)
    print(f"\nFile extensions: {dict(ext_counts)}")

    # Check for naming patterns
    names = [p.stem for p in images[:10]]
    print(f"\nFirst 10 filenames: {names}")

    print(f"\n{'='*60}")
    print("EDA done. Now go train.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
