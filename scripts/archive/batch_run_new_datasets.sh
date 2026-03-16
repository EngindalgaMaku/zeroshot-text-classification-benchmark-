#!/bin/bash
# Batch script to run all new dataset experiments (IMDB + SST-2)
# Total: 14 experiments (7 models × 2 datasets)
# Estimated time: 1-2 hours

echo "================================================================================"
echo "Running New Dataset Experiments (IMDB + SST-2)"
echo "================================================================================"
echo ""
echo "Total experiments: 14 (7 models × 2 datasets)"
echo "Estimated time: 1-2 hours"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

echo ""
echo "================================================================================"
echo "IMDB Experiments (7 models)"
echo "================================================================================"
echo ""

echo "[1/14] Running IMDB with MPNet..."
python main.py --config experiments/exp_imdb_mpnet.yaml || echo "ERROR: IMDB MPNet failed"

echo "[2/14] Running IMDB with Qwen3..."
python main.py --config experiments/exp_imdb_qwen3.yaml || echo "ERROR: IMDB Qwen3 failed"

echo "[3/14] Running IMDB with Snowflake..."
python main.py --config experiments/exp_imdb_snowflake.yaml || echo "ERROR: IMDB Snowflake failed"

echo "[4/14] Running IMDB with Instructor..."
python main.py --config experiments/exp_imdb_instructor.yaml || echo "ERROR: IMDB Instructor failed"

echo "[5/14] Running IMDB with Jina-v3..."
python main.py --config experiments/exp_imdb_jina_v3.yaml || echo "ERROR: IMDB Jina-v3 failed"

echo "[6/14] Running IMDB with BGE-M3..."
python main.py --config experiments/exp_imdb_bge.yaml || echo "ERROR: IMDB BGE failed"

echo "[7/14] Running IMDB with E5-large..."
python main.py --config experiments/exp_imdb_e5.yaml || echo "ERROR: IMDB E5 failed"

echo ""
echo "================================================================================"
echo "SST-2 Experiments (7 models)"
echo "================================================================================"
echo ""

echo "[8/14] Running SST-2 with MPNet..."
python main.py --config experiments/exp_sst2_mpnet.yaml || echo "ERROR: SST-2 MPNet failed"

echo "[9/14] Running SST-2 with Qwen3..."
python main.py --config experiments/exp_sst2_qwen3.yaml || echo "ERROR: SST-2 Qwen3 failed"

echo "[10/14] Running SST-2 with Snowflake..."
python main.py --config experiments/exp_sst2_snowflake.yaml || echo "ERROR: SST-2 Snowflake failed"

echo "[11/14] Running SST-2 with Instructor..."
python main.py --config experiments/exp_sst2_instructor.yaml || echo "ERROR: SST-2 Instructor failed"

echo "[12/14] Running SST-2 with Jina-v3..."
python main.py --config experiments/exp_sst2_jina_v3.yaml || echo "ERROR: SST-2 Jina-v3 failed"

echo "[13/14] Running SST-2 with BGE-M3..."
python main.py --config experiments/exp_sst2_bge.yaml || echo "ERROR: SST-2 BGE failed"

echo "[14/14] Running SST-2 with E5-large..."
python main.py --config experiments/exp_sst2_e5.yaml || echo "ERROR: SST-2 E5 failed"

echo ""
echo "================================================================================"
echo "All Experiments Complete!"
echo "================================================================================"
echo ""
echo "Results saved to: results/raw/"
echo ""
echo "Next steps:"
echo "1. Verify all result files exist"
echo "2. Update results database"
echo "3. Regenerate visualizations"
echo ""
