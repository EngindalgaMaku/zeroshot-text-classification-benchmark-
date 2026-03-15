content = open('src/runner.py', encoding='utf-8').read(); content = content.replace('model_type = \
reranker\', ''); content = content.replace('model_type = \biencoder\', ''); content = content.replace('metrics[model_type]', 'metrics[\biencoder\]'); open('src/runner.py', 'w', encoding='utf-8').write(content); print('Done')
