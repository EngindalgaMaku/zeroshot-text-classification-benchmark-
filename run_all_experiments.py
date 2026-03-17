"""Run all L1/L2/L3 experiments sequentially and collect results."""
import subprocess
import sys
import json
import os
from pathlib import Path

EXPERIMENTS = [
    # ag_news
    "experiments/exp_ag_news_mpnet_name_only.yaml",
    "experiments/exp_ag_news_mpnet_l2_anchored.yaml",
    "experiments/exp_ag_news_mpnet_l3_anchored.yaml",
    # banking77
    "experiments/exp_banking77_mpnet_name_only.yaml",
    "experiments/exp_banking77_mpnet_l2_anchored.yaml",
    "experiments/exp_banking77_mpnet_l3_anchored.yaml",
    # dbpedia_14
    "experiments/exp_dbpedia_14_mpnet_name_only.yaml",
    "experiments/exp_dbpedia_14_mpnet_l2_anchored.yaml",
    "experiments/exp_dbpedia_14_mpnet_l3_anchored.yaml",
    # yahoo_answers_topics
    "experiments/exp_yahoo_answers_topics_mpnet_name_only.yaml",
    "experiments/exp_yahoo_answers_topics_mpnet_l2_anchored.yaml",
    "experiments/exp_yahoo_answers_topics_mpnet_l3_anchored.yaml",
    # 20newsgroups
    "experiments/exp_20newsgroups_mpnet_name_only.yaml",
    "experiments/exp_20newsgroups_mpnet_l2_anchored.yaml",
    "experiments/exp_20newsgroups_mpnet_l3_anchored.yaml",
    # imdb
    "experiments/exp_imdb_mpnet_name_only.yaml",
    "experiments/exp_imdb_mpnet_l2_anchored.yaml",
    "experiments/exp_imdb_mpnet_l3_anchored.yaml",
    # sst2
    "experiments/exp_sst2_mpnet_name_only.yaml",
    "experiments/exp_sst2_mpnet_l2_anchored.yaml",
    "experiments/exp_sst2_mpnet_l3_anchored.yaml",
    # twitter-financial
    "experiments/exp_twitter_financial_mpnet_name_only.yaml",
    "experiments/exp_twitter_financial_mpnet_l2_anchored.yaml",
    "experiments/exp_twitter_financial_mpnet_l3_anchored.yaml",
    # go_emotions
    "experiments/exp_go_emotions_mpnet_name_only.yaml",
    "experiments/exp_go_emotions_mpnet_l2_anchored.yaml",
    "experiments/exp_go_emotions_mpnet_l3_anchored.yaml",
]

errors = []

for config in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"Running: {config}")
    print('='*60)
    result = subprocess.run(
        [sys.executable, "main.py", "--config", config, "--skip-existing"],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: {config} failed with return code {result.returncode}")
        errors.append(config)
    else:
        print(f"OK: {config}")

print(f"\n{'='*60}")
print(f"Done. Errors: {len(errors)}")
for e in errors:
    print(f"  FAILED: {e}")
