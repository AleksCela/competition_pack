# Competition Pack — PlantVillage Image Classification

## What is this?

A self-contained kit for a plant disease image classification competition.
You train a **Vision Transformer (ViT)** on labeled plant leaf images, then
predict labels for unlabeled test images and submit a CSV.

**Model:** `google/vit-base-patch16-224` (pretrained, fine-tuned on PlantVillage)
**Dataset:** PlantVillage — 38 classes of healthy/diseased plant leaves
**Output:** `submission.csv` with columns `image_id,label`

---

## Files at a glance

| File | What it does |
|---|---|
| `train_vit.py` | Trains the ViT model in 2 phases (frozen backbone, then full fine-tune) |
| `predict_test.py` | Runs inference on test images, writes `submission.csv` |
| `final_submission_check.py` | Validates your CSV before you upload it |
| `make_mock_test_set.py` | Samples training images into a fake test set for local testing |
| `requirements.txt` | Python dependencies (torch installed separately via setup scripts) |
| `setup_venv_windows.ps1` | One-command venv + deps setup on Windows |
| `setup_venv_linux.sh` | One-command venv + deps setup on Linux/RunPod |
| `run_predict_and_check.ps1` | Windows: predict then validate in one shot |
| `run_predict_and_check.sh` | Linux: predict then validate in one shot |
| `runpod_setup.txt` | Step-by-step RunPod GPU setup instructions |

---

## Typical workflow

### Scenario A: Train on GPU (RunPod), predict locally

1. **On RunPod** — follow `runpod_setup.txt` or:
   ```bash
   bash setup_venv_linux.sh
   .venv/bin/python train_vit.py --max-images 0   # 0 = use all images
   ```
2. **Download** the model folder `vit-model/phase2_finetuned/` to your laptop.
3. **On laptop** — put test images in `data/test/`, then:
   ```
   # Windows
   powershell -ExecutionPolicy Bypass -File .\run_predict_and_check.ps1 -ModelDir vit-model/phase2_finetuned -TestDir data/test

   # Linux/Mac
   bash run_predict_and_check.sh vit-model/phase2_finetuned data/test
   ```
4. **Upload** `submission.csv`.

### Scenario B: Everything on one machine

```bash
bash setup_venv_linux.sh
.venv/bin/python train_vit.py --max-images 0
.venv/bin/python predict_test.py --model-dir vit-model/phase2_finetuned --test-dir data/test
.venv/bin/python final_submission_check.py --csv submission.csv --test-dir data/test --model-dir vit-model/phase2_finetuned
```

---

## How training works (train_vit.py)

The script does **two-phase transfer learning**:

1. **Phase 1 — Linear probe** (2 epochs, lr=1e-3): Freezes the ViT backbone,
   only trains the new 38-class classification head. Gets a decent baseline fast.

2. **Phase 2 — Full fine-tune** (5 epochs, lr=2e-5): Unfreezes everything,
   fine-tunes the whole model with a lower learning rate. This is where the
   real accuracy gains happen.

Both phases save to `vit-model/`. The final model you want is
**`vit-model/phase2_finetuned/`**.

**Key flags:**
- `--max-images 10000` — cap dataset size for faster iteration (default)
- `--max-images 0` — use the full dataset (recommended for final submission)

**Data expectations:**
- Training images should be in `data/train/` in ImageFolder format
  (subdirectories named by class, images inside each)
- Only RGB images are used (grayscale/RGBA are filtered out automatically)

---

## How prediction works (predict_test.py)

- Loads the trained model from `--model-dir`
- Reads all images from `--test-dir`, converts to RGB
- Runs batched inference, writes `image_id,label` rows to CSV
- Works on CPU (just slower) — use `--batch-size 4` if memory is tight

---

## How validation works (final_submission_check.py)

Checks your CSV for common mistakes before you upload:
- Header is exactly `image_id,label`
- One row per test image, no duplicates, no missing
- Labels match the model's known classes

---

## Quick local test (no real test set yet?)

Use `make_mock_test_set.py` to create a fake test set from training data:

```bash
python make_mock_test_set.py --train-dir data/train --out-dir data/test_mock --num-images 200
python predict_test.py --model-dir vit-model/phase2_finetuned --test-dir data/test_mock
python final_submission_check.py --csv submission.csv --test-dir data/test_mock --model-dir vit-model/phase2_finetuned
```

This also saves `data/test_mock_labels.csv` so you can score locally.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| OOM during training | Lower `per_device_train_batch_size` in `train_vit.py` (try 2) |
| OOM during prediction | Use `--batch-size 2` or `--batch-size 1` |
| "No images found" | Check your `--test-dir` path; images must be directly in that folder |
| CUDA not available | Prediction still works on CPU, just slower. Training will be very slow without GPU |
| Processor config not found | Script auto-falls back to `google/vit-base-patch16-224` processor |

---

## What to ask Claude tomorrow

If you're retraining or doing something different, here are good starting prompts:

- "I need to retrain with the full dataset on RunPod, walk me through it"
- "I want to try a different model architecture"
- "Help me tune hyperparameters for better accuracy"
- "I have the real test set now, help me generate the submission"
- "I want to add data augmentation to improve generalization"
