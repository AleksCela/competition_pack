# Competition Day — 3 Hour Sprint

**3 hours. 3 submissions. Win.**

---

## T+0 — Deploy RunPod (2 min)

- Open RunPod template link from competition page
- Pick **A100 80GB** or **H100** — biggest GPU available
- Deploy, wait ~2 min, click **Connect → JupyterLab**

---

## T+2 — Open TWO terminal tabs immediately

Do these **in parallel** — don't wait for one to finish before starting the other.

**Tab 1 — Install deps:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets huggingface_hub accelerate scikit-learn pillow timm
huggingface-cli login
# paste your HF token
```

**Tab 2 — Download data (paste your token first if it asks):**
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('SmellsLikeAISpirit/plant-disease-train', repo_type='dataset', local_dir='/workspace/data/train')
snapshot_download('SmellsLikeAISpirit/plant-disease-test',  repo_type='dataset', local_dir='/workspace/data/test')
print('Data ready.')
"
```

While both run, upload your competition pack via JupyterLab drag-and-drop into `/workspace/`.

---

## T+15 — SUBMISSION 1: CLIP zero-shot

**Do this the moment data finishes downloading.**

```bash
cd /workspace/competition_pack/advanced

python clip_zero_shot.py \
  --test-dir /workspace/data/test \
  --output-csv /workspace/submission.csv

python ../final_submission_check.py \
  --csv /workspace/submission.csv \
  --test-dir /workspace/data/test
```

Download `submission.csv` → **submit immediately**. You're on the board. Safety net secured.

---

## T+25 — MAIN MODEL: Train SigLIP 2

**This is your real weapon.**

```bash
cd /workspace/competition_pack/advanced

python train_vit_large.py \
  --model google/siglip2-so400m-patch14-384 \
  --data-dir /workspace/data/train \
  --max-images 0 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --epochs 5 \
  --lr 1e-5 \
  --weighted-loss \
  --no-phase1 \
  --output-dir /workspace/siglip2-model
```

> `--no-phase1` skips the linear probe warmup and goes straight to full fine-tuning.
> Saves ~20-30 min. Accuracy loss is negligible with a pretrained model this strong.
> `--weighted-loss` is non-negotiable — protects the `other` class.

**Expected time on A100 80GB**: ~50-60 min
**Expected time on H100**: ~35-40 min
**Expected val accuracy**: 93-96%

---

## T+1h25 — SUBMISSION 2: SigLIP 2 TTA + OOD protection

Once training finishes:

```bash
cd /workspace/competition_pack/advanced

python predict_tta.py \
  --model-dir /workspace/siglip2-model/best \
  --test-dir /workspace/data/test \
  --output-csv /workspace/submission_siglip2.csv \
  --num-tta 6 \
  --ood-threshold 0.5

python ../final_submission_check.py \
  --csv /workspace/submission_siglip2.csv \
  --test-dir /workspace/data/test

cp /workspace/submission_siglip2.csv /workspace/submission.csv
```

Download → **submit**. This is likely your winning submission.

> `--ood-threshold 0.5`: any image the model is less than 50% confident about gets classified as `other`.
> Plant disease images are typically 95%+ confident. Potholes are not.

---

## T+1h40 — SECOND MODEL: Train DINOv2

Start immediately after submitting. Different architecture = different errors = better ensemble.

```bash
cd /workspace/competition_pack/advanced

python train_dinov2.py \
  --model facebook/dinov2-large \
  --data-dir /workspace/data/train \
  --max-images 0 \
  --batch-size 32 \
  --epochs 5 \
  --lr 5e-6 \
  --grad-accum 2 \
  --output-dir /workspace/dinov2-model
```

**Expected time on A100 80GB**: ~55-65 min

---

## T+2h40 — SUBMISSION 3: Ensemble

```bash
cd /workspace/competition_pack/advanced

python ensemble_predict.py \
  --model-dirs /workspace/siglip2-model/best /workspace/dinov2-model/best \
  --model-types hf dinov2 \
  --weights 0.6 0.4 \
  --test-dir /workspace/data/test \
  --output-csv /workspace/submission_ensemble.csv \
  --ood-threshold 0.5

python ../final_submission_check.py \
  --csv /workspace/submission_ensemble.csv \
  --test-dir /workspace/data/test

cp /workspace/submission_ensemble.csv /workspace/submission.csv
```

Download → **submit**.

---

## T+3h — Lock in your best before 16:30

Go to the leaderboard. Pick your highest scoring submission as **final**.
Do NOT submit anything new in the last 15 minutes.

---

## GPU batch size cheat sheet

| GPU | SigLIP 2 batch | DINOv2 batch |
|---|---|---|
| A100 40GB | 32 | 16 |
| A100 80GB | 64 | 32 |
| H100 80GB | 96 | 48 |
| RTX 6000 48GB | 48 | 24 |

If you get OOM: halve `--batch-size` and double `--grad-accum`.

---

## If DINOv2 isn't done in time

Skip the ensemble. Keep Submission 2 (SigLIP 2) as your final.
Do NOT rush a half-trained ensemble — it will be worse than a fully trained single model.

---

## If SigLIP 2 isn't on HuggingFace yet

Fall back to ViT-large — same command, different model:

```bash
python train_vit_large.py \
  --model google/vit-large-patch16-224 \
  --data-dir /workspace/data/train \
  --max-images 0 \
  --batch-size 64 \
  --epochs 5 \
  --lr 2e-5 \
  --weighted-loss \
  --no-phase1 \
  --output-dir /workspace/vitlarge-model
```

---

## Submission format

```
id,label
img_36ef94e7.jpg,Tomato___Tomato_Yellow_Leaf_Curl_Virus
img_59ebf94f.jpg,other
```

- File must be named `submission.csv`
- Header exactly: `id,label`
- Exactly **10,976 rows**
- Labels are case-sensitive

---

## 3-hour timeline at a glance

```
T+0:00  Deploy pod
T+0:02  Start: pip install + data download (parallel tabs)
T+0:15  CLIP zero-shot → Submission 1
T+0:25  Start SigLIP 2 training
T+1:25  SigLIP 2 done → TTA predict → Submission 2
T+1:40  Start DINOv2 training
T+2:40  DINOv2 done → Ensemble → Submission 3
T+3:00  Select final. Done.
```
