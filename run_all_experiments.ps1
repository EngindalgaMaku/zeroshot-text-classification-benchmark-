# Run all label description experiments using conda environment
# Usage: .\run_all_experiments.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "LABEL DESCRIPTION EXPERIMENTS - BATCH RUNNER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$condaEnv = "zeroshot"

# Check GPU using conda run
Write-Host "Checking GPU in conda environment '$condaEnv'..." -ForegroundColor Green
conda run -n $condaEnv python -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
Write-Host ""

# Create results directory
$resultsDir = "results\full_label_descriptions"
if (!(Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
}

# Run all experiments using conda run
Write-Host "Starting experiments... This will take several hours." -ForegroundColor Green
Write-Host "Results will be saved to: $resultsDir" -ForegroundColor Yellow
Write-Host "Progress log: results\experiment_run.log" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop (you can resume later by running this script again)" -ForegroundColor Red
Write-Host ""
Start-Sleep -Seconds 3

# Run the Python script via conda
conda run -n $condaEnv --live-stream python scripts\run_all_label_descriptions.py

# Keep window open
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RUN COMPLETED - Press any key to close" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
