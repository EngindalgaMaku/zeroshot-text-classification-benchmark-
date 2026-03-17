@echo off
chcp 65001 >nul
echo ============================================================
echo LABEL DESCRIPTION EXPERIMENTS - BATCH RUNNER
echo ============================================================
echo.

:: Activate conda environment
call conda activate zeroshot
if errorlevel 1 (
    echo ERROR: Could not activate conda environment 'zeroshot'
    echo Make sure conda is installed and 'zeroshot' environment exists
    pause
    exit /b 1
)

echo ✅ Conda environment 'zeroshot' activated
echo.

:: Check GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'); print('CUDA:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo.

:: Create results directory if not exists
if not exist "results\full_label_descriptions" mkdir "results\full_label_descriptions"

:: Run all experiments
echo Starting experiments... This will take several hours.
echo Results will be saved to: results\full_label_descriptions\
echo Progress log: results\experiment_run.log
echo Failed experiments: results\failed_experiments.json
echo.
echo Press Ctrl+C to stop (you can resume later by running this script again)
echo.
timeout /t 5 >nul

python scripts\run_all_label_descriptions.py

:: Keep window open to see final results
echo.
echo ============================================================
echo RUN COMPLETED - Press any key to close
echo ============================================================
pause >nul
