@echo off
call conda activate zeroshot

echo ==========================================
echo PROMPT ENGINEERING TEST SUITE
echo Testing Manual vs V4 LLM Descriptions
echo ==========================================
echo.

echo ==========================================
echo TEST 1/3: L1 (name_only - Baseline)
echo Expected F1: ~50%%
echo ==========================================
python main.py --config scripts\prompt_engineering\configs\ag_news_instructor_l1_name_only.yaml
echo.
echo Test 1 Complete!
echo.

echo ==========================================
echo TEST 2/3: Manual (description - Gold Standard)
echo Expected F1: ~64%%
echo ==========================================
python main.py --config scripts\prompt_engineering\configs\ag_news_instructor_manual_descriptions.yaml
echo.
echo Test 2 Complete!
echo.

echo ==========================================
echo TEST 3/3: V4 LLM (Manual-Inspired Prompt)
echo Expected F1: ~62-64%% (Hope!)
echo ==========================================
python main.py --config scripts\prompt_engineering\configs\ag_news_instructor_v4_llm_descriptions.yaml
echo.
echo Test 3 Complete!
echo.

echo ==========================================
echo ALL TESTS COMPLETE!
echo ==========================================
echo.
echo Results saved in: scripts\prompt_engineering\results\
echo.
echo Next steps:
echo 1. Compare metrics files
echo 2. If V4 ≈ Manual, use V4 prompt for all datasets!
echo 3. If V4 still low, iterate on prompt
echo.
pause