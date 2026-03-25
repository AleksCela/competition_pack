param(
    [string]$ModelDir = "vit-model/phase2_finetuned",
    [string]$TestDir = "data/test",
    [string]$OutCsv = "submission.csv",
    [int]$BatchSize = 4,
    [string]$ProcessorName = "google/vit-base-patch16-224"
)

if (-Not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Error "Missing .venv. Run setup_venv_windows.ps1 first."
    exit 1
}

.\.venv\Scripts\python.exe predict_test.py --model-dir $ModelDir --test-dir $TestDir --output-csv $OutCsv --batch-size $BatchSize --processor-name $ProcessorName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.\.venv\Scripts\python.exe final_submission_check.py --csv $OutCsv --test-dir $TestDir --model-dir $ModelDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done. Submission file ready:" $OutCsv
