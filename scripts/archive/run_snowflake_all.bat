@echo off
REM Run all Snowflake Arctic Embed experiments
REM Uses --skip-existing to avoid re-running completed experiments

echo ========================================
echo Running Snowflake Arctic Embed Experiments
echo ========================================
echo.

echo [1/4] Running AG News...
python main.py --config experiments/exp_agnews_snowflake.yaml --skip-existing
echo.

echo [2/4] Running Banking77...
python main.py --config experiments/exp_banking77_snowflake.yaml --skip-existing
echo.

echo [3/4] Running DBPedia-14...
python main.py --config experiments/exp_dbpedia_snowflake.yaml --skip-existing
echo.

echo [4/4] Running 20 Newsgroups...
python main.py --config experiments/exp_20newsgroups_snowflake.yaml --skip-existing
echo.

echo ========================================
echo All Snowflake experiments completed!
echo ========================================
echo.
echo To generate updated visualizations, run:
echo   python scripts/generate_beautiful_plots.py
echo   python scripts/generate_heatmap_report.py
