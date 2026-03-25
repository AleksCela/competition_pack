# COMPETITION DAY GAMEPLAN

You have ~6 hours. Here's the exact order. Do NOT skip steps.

## Phase 0 — Setup (first 20 minutes)
```bash
# On RunPod (or wherever your GPU is):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets huggingface_hub accelerate scikit-learn pillow timm open_clip_torch
huggingface-cli login   # paste your token

# Look at the data
python eda.py --data-dir data/test
# This tells you: how many images, any weird formats, class distribution if labeled
```

## Phase 1 — CLIP Zero-Shot (T+20min → T+1:00)
**Goal: Get on the leaderboard NOW. No training needed.**
```bash
python clip_zero_shot.py --test-dir data/test --output-csv submission_clip.csv
```
Upload `submission_clip.csv` immediately. Expected: 60-75%.
This is your safety net — you have a score no matter what.

## Phase 2 — Fine-Tune Main Model (T+1:00 → T+3:00)
**Goal: Real accuracy. Pick ONE of these based on your GPU:**

| GPU VRAM | Model | Script |
|---|---|---|
| 40GB+ (A100) | vit-large-patch16-224 | `train_vit_large.py` |
| 16-24GB | vit-base-patch16-224 | `../train_vit.py` (original) |
| 8GB | vit-base + gradient checkpointing | `train_vit_large.py --model google/vit-base-patch16-224 --batch-size 4` |

```bash
# A100 example — full dataset, vit-large:
python train_vit_large.py --data-dir data/train --max-images 0 --model google/vit-large-patch16-224

# Then predict:
python predict_tta.py --model-dir vit-model/best --test-dir data/test --output-csv submission_vit.csv
```
Upload `submission_vit.csv`. Expected: 88-95%.

## Phase 3 — Analyze + Fix Weak Spots (T+3:00 → T+4:00)
```bash
# See where your model is confused (uses validation set):
python analyze_confusion.py --model-dir vit-model/best --data-dir data/train

# Look at the output — it tells you which classes are confused
# Common: Tomato diseases look alike, Apple scab vs Cedar rust
```
If time allows, retrain with `--weighted-loss` flag to boost weak classes.

## Phase 4 — Second Model + Ensemble (T+4:00 → T+5:30)
```bash
# Train a second architecture (DINOv2 or Swin):
python train_dinov2.py --data-dir data/train --max-images 0

# Ensemble both models:
python ensemble_predict.py \
    --model-dirs vit-model/best dinov2-model/best \
    --test-dir data/test \
    --output-csv submission_ensemble.csv
```
Upload `submission_ensemble.csv`. Expected: +1-3% over single model.

## Phase 5 — Lock It (T+5:30 → T+6:00)
- **STOP TRAINING.**
- Check leaderboard. Select your best-scoring submission as final.
- Verify the selected submission is correct.
- Do NOT submit anything new in the last 30 minutes unless you're confident.

---

## Emergency Shortcuts
- **"I only have 2 hours left"** → Skip ensemble. Just do CLIP zero-shot + one fine-tune + TTA.
- **"Training is too slow"** → Use `--max-images 10000` to cap dataset. Still gets 88%+.
- **"OOM error"** → Halve batch size. Add `--grad-accum 4`. Use fp16 (on by default).
- **"Test set looks different from PlantVillage"** → CLIP zero-shot might beat fine-tuned models. Submit both, compare.

## Quick Reference — Submission Format
```
image_id,label
img_001.jpg,Tomato___Late_blight
img_002.jpg,Apple___healthy
```
Exact header. Case sensitive. One row per test image. No duplicates.
