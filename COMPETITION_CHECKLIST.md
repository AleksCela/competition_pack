Competition Pack Checklist

Goal
- Train on GPU machine if needed.
- Predict on local machine and submit CSV.

Folder contents
- train_vit.py
- predict_test.py
- final_submission_check.py
- requirements.txt
- setup_venv_windows.ps1
- setup_venv_linux.sh
- run_predict_and_check.ps1
- run_predict_and_check.sh

Windows quickstart
1) Open PowerShell in this folder.
2) Run:
   powershell -ExecutionPolicy Bypass -File .\setup_venv_windows.ps1
3) Put test images in data/test (or pass a custom path).
4) Run prediction + validation:
   powershell -ExecutionPolicy Bypass -File .\run_predict_and_check.ps1 -ModelDir vit-model/phase2_finetuned -TestDir data/test -OutCsv submission.csv -BatchSize 4
5) Upload submission.csv.

RunPod / Linux quickstart
1) In terminal at this folder:
   bash setup_venv_linux.sh
2) Train if needed:
   .venv/bin/python train_vit.py --max-images 0
3) Predict + validate:
   bash run_predict_and_check.sh vit-model/phase2_finetuned data/test submission.csv 4
4) Upload submission.csv.

If no GPU is available
- Prediction still works on CPU.
- Use smaller batch size (2 or 4).

What to transfer to laptop at minimum
- predict_test.py
- final_submission_check.py
- requirements.txt
- your model folder: vit-model/phase2_finetuned (final files only)

Pre-submit checks
- CSV header exactly: image_id,label
- One row per test image
- No duplicate image_id rows
- Labels match model classes
