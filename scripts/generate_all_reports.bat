@echo off
echo ======================================================================
echo Generating All Reports
echo ======================================================================
echo.

echo [1/4] Generating Heatmap Report (PDF)...
python scripts/generate_heatmap_report.py
if %errorlevel% neq 0 (
    echo ERROR: Heatmap report failed
    pause
    exit /b 1
)
echo.

echo [2/4] Generating Tables and Plots...
python scripts/generate_tables_and_plots.py
if %errorlevel% neq 0 (
    echo ERROR: Tables and plots failed
    pause
    exit /b 1
)
echo.

echo [3/4] Generating Beautiful Plots...
python scripts/generate_beautiful_plots.py
if %errorlevel% neq 0 (
    echo ERROR: Beautiful plots failed
    pause
    exit /b 1
)
echo.

echo [4/4] Generating Dataset Report...
python scripts/generate_dataset_report.py
if %errorlevel% neq 0 (
    echo ERROR: Dataset report failed
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo All Reports Generated Successfully!
echo ======================================================================
echo.
echo Check these locations:
echo   - reports/F1_HEATMAP_PUBLICATION.pdf
echo   - results/plots/
echo   - results/tables/
echo.
pause
