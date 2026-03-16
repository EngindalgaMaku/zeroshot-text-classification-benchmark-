@echo off
echo Running Jina with task="classification" on all 6 datasets...
echo.

python main.py --config experiments/ag_news_jina_task.yaml
python main.py --config experiments/SetFit_20_newsgroups_jina_task.yaml
python main.py --config experiments/dbpedia_14_jina_task.yaml
python main.py --config experiments/banking77_jina_task.yaml
python main.py --config experiments/yahoo_answers_topics_jina_task.yaml
python main.py --config experiments/zeroshot_twitter_financial_news_sentiment_jina_task.yaml

echo.
echo All Jina task="classification" experiments complete!
pause