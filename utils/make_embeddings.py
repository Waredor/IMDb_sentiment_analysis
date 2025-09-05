import os
import ast
import logging
import pickle
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
import yaml
from gensim.models import Word2Vec

def get_project_root(start_path):
    current = start_path
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Project root was not found")

def parse_tokens(tokens):
    if isinstance(tokens, str):
        try:
            return ast.literal_eval(tokens)
        except (ValueError, SyntaxError):
            return tokens.split()
    return tokens

if __name__ == '__main__':
    init_path = os.path.abspath(__file__)
    PROJECT_ROOT_PATH = get_project_root(init_path)
    LOGGER_PATH = os.path.join(PROJECT_ROOT_PATH, 'logs', 'make_embeddings_log.txt')

    logger = logging.getLogger('make_embeddings_logger')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(LOGGER_PATH, maxBytes=1048576, backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('configuring paths')
    PIPELINE_CONFIG_PATH = os.path.join(PROJECT_ROOT_PATH, 'configs', 'project_config.yaml')
    EMBEDDINGS_DICT_PATH = os.path.join(PROJECT_ROOT_PATH, 'data', 'token_embeddings.pkl')

    with open(PIPELINE_CONFIG_PATH, mode='r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    DATA_DIR = os.path.join(PROJECT_ROOT_PATH, 'data', 'IMDB Dataset_tokenized.csv')

    logger.info('reading dataframe')
    data = pd.read_csv(DATA_DIR, encoding='utf-8')
    data['tokens'] = data['tokens'].apply(parse_tokens)

    sentences = data['tokens'].tolist()
    unique_tokens = list(set(token for sublist in sentences for token in sublist))

    logger.info('train Word2Vec model')
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=5,  # Игнорируем редкие токены (уже заменены на <UNK>)
        workers=4,
        sg=1,
        epochs=10
    )

    logger.info('get token embeddings')
    token_embeddings = {}
    vector_size = model.vector_size

    for token in unique_tokens:
        if token in model.wv:
            token_embeddings[token] = model.wv[token]
        else:
            token_embeddings[token] = model.wv['<UNK>']  # Используем эмбеддинг <UNK> для отсутствующих токенов

    # Добавляем <PAD> и <UNK> в эмбеддинги
    token_embeddings['<PAD>'] = np.zeros(vector_size)
    token_embeddings['<UNK>'] = model.wv['<UNK>'] if '<UNK>' in model.wv else np.random.normal(0, 0.1, vector_size)

    logger.info(f"Vocab size: {len(token_embeddings)}")
    logger.info('save embeddings dict')
    with open(EMBEDDINGS_DICT_PATH, 'wb') as f:
        pickle.dump(token_embeddings, f)

    logger.info('pipeline finished successfully')