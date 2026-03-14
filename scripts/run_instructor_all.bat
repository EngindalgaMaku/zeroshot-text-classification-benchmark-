@echo off
echo Running INSTRUCTOR on all 6 datasets...
echo.

python main.py --config experiments/ag_news_instructor.yaml --skip-existing
python main.py --config experiments/SetFit_20_newsgroups_instructor.yaml --skip-existing
python main.py --config experiments/dbpedia_14_instructor.yaml --skip-existing
python main.py --config experiments/banking77_instructor.yaml --skip-existing
python main.py --config experiments/yahoo_answers_topics_instructor.yaml --skip-existing
python main.py --config experiments/zeroshot_twitter_financial_news_sentiment_instructor.yaml --skip-existing

echo.
echo All INSTRUCTOR experiments complete!
pause