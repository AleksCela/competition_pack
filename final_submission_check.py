import argparse
import csv
from pathlib import Path


def list_image_ids(test_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p.name for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def load_allowed_labels(model_dir: Path):
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_dir)
        id2label = getattr(config, "id2label", None)
        if not id2label:
            return None
        return {str(v) for v in id2label.values()}
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Validate submission CSV before upload.")
    parser.add_argument("--csv", default="submission.csv", help="Submission CSV path")
    parser.add_argument("--test-dir", default="data/test", help="Test directory used for prediction")
    parser.add_argument("--model-dir", default="vit-model/phase2_finetuned", help="Optional model directory for label validation")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    test_dir = Path(args.test_dir)
    model_dir = Path(args.model_dir)

    errors = []

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if fieldnames != ["image_id", "label"]:
            errors.append(f"Header must be exactly: image_id,label. Found: {fieldnames}")

        rows = list(reader)

    if not rows:
        errors.append("CSV has no prediction rows.")

    image_ids = [r.get("image_id", "") for r in rows]
    labels = [r.get("label", "") for r in rows]

    if any(not x for x in image_ids):
        errors.append("Found empty image_id values.")
    if any(not x for x in labels):
        errors.append("Found empty label values.")

    duplicates = len(image_ids) - len(set(image_ids))
    if duplicates > 0:
        errors.append(f"Found duplicate image_id rows: {duplicates}")

    if test_dir.exists():
        expected_ids = list_image_ids(test_dir)
        expected_set = set(expected_ids)
        got_set = set(image_ids)

        missing = sorted(expected_set - got_set)
        extra = sorted(got_set - expected_set)

        if missing:
            errors.append(f"Missing predictions for {len(missing)} test images.")
        if extra:
            errors.append(f"CSV contains {len(extra)} image_id values not found in test_dir.")

        if len(rows) != len(expected_ids):
            errors.append(
                f"Row count mismatch. CSV rows: {len(rows)} vs test images: {len(expected_ids)}"
            )

    if model_dir.exists():
        allowed_labels = load_allowed_labels(model_dir)
        if allowed_labels is not None:
            bad_labels = sorted({lab for lab in labels if lab not in allowed_labels})
            if bad_labels:
                errors.append(
                    f"Found labels not in model config id2label: {bad_labels[:5]}"
                )

    if errors:
        print("Submission validation FAILED:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("Submission validation PASSED.")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
