@echo off
echo Running all nomic-embed experiments...
echo.
python main.py --config experiments/exp_banking77_nomic.yaml
python main.py --config experiments/exp_dbpedia_nomic.yaml
python main.py --config experiments/exp_goemotions_nomic.yaml
python main.py --config experiments/exp_imdb_nomic.yaml
python main.py --config experiments/exp_sst2_nomic.yaml
python main.py --config experiments/exp_20newsgroups_nomic.yaml
python main.py --config experiments/exp_yahoo_answers_nomic.yaml
python main.py --config experiments/exp_twitter_financial_nomic.yaml
echo.
echo All experiments complete!
