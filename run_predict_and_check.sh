#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-vit-model/phase2_finetuned}"
TEST_DIR="${2:-data/test}"
OUT_CSV="${3:-submission.csv}"
BATCH_SIZE="${4:-4}"
PROCESSOR_NAME="${5:-google/vit-base-patch16-224}"

if [[ ! -f .venv/bin/python ]]; then
  echo "Missing .venv. Run setup_venv_linux.sh first."
  exit 1
fi

.venv/bin/python predict_test.py --model-dir "$MODEL_DIR" --test-dir "$TEST_DIR" --output-csv "$OUT_CSV" --batch-size "$BATCH_SIZE" --processor-name "$PROCESSOR_NAME"
.venv/bin/python final_submission_check.py --csv "$OUT_CSV" --test-dir "$TEST_DIR" --model-dir "$MODEL_DIR"

echo "Done. Submission file ready: $OUT_CSV"
