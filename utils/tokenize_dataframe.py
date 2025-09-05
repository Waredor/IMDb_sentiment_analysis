import os
import logging
import re
from logging.handlers import RotatingFileHandler
import pandas as pd
import nltk
import yaml
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

def tokenize_text_on_words(text: str, min_freq=5) -> list:
    global stop_words, lemmatizer, token_counter

    # Словарь для обработки контракций
    contractions = {
        "n\'t": "not",
        "\'s": "is",
        "\'ve": "have",
        "\'m": "am",
        "\'re": "are",
        "\'ll": "will",
        "\'d": "would"
    }

    # Токенизация
    tokens = word_tokenize(text.lower())

    cleaned = []
    for token in tokens:
        # Обработка контракций
        token = contractions.get(token, token)
        # Удаление HTML-тегов
        token = re.sub(r'<[^>]+>', '', token)
        # Удаление специальных символов, кроме букв, чисел и дефисов
        token = re.sub(r'[^\w\s-]', '', token)
        # Сохранение чисел, если они значимы (например, "10/10")
        token = token.strip()
        if token and (token in stop_words or not token.isalnum()):
            continue  # Пропускаем стоп-слова и неалфавитно-цифровые токены
        # Лемматизация
        token = lemmatizer.lemmatize(token)
        if token:
            cleaned.append(token)
            token_counter[token] += 1


    cleaned = [token if token_counter[token] >= min_freq else '<UNK>' for token in cleaned]
    return cleaned

def get_project_root(start_path):
    current = start_path
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Project root was not found")

if __name__ == '__main__':
    init_path = os.path.abspath(__file__)
    PROJECT_ROOT_PATH = get_project_root(init_path)
    LOGGER_PATH = os.path.join(PROJECT_ROOT_PATH, 'logs', 'tokenize_dataframe_log.txt')

    logger = logging.getLogger('tokenize-dataframe_logger')
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

    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')  # Для лемматизации

    logger.info('configuring paths')
    PIPELINE_CONFIG_PATH = os.path.join(PROJECT_ROOT_PATH, 'configs', 'project_config.yaml')

    with open(PIPELINE_CONFIG_PATH, mode='r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    DATA_PATH = yaml_data['raw_data_path']
    DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT_PATH, 'data')
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    DATA_OUTPUT_PATH = os.path.join(DATA_OUTPUT_DIR, 'IMDB Dataset_tokenized.csv')

    logger.info('creating dataframe')
    data = pd.read_csv(DATA_PATH, encoding='utf-8')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    token_counter = Counter()

    logger.info('tokenize text data')

    for text in data['review']:
        tokenize_text_on_words(text)
    logger.info(f"Initial vocab size: {len(token_counter)}")


    data['tokens'] = data['review'].apply(lambda x: tokenize_text_on_words(x, min_freq=5))
    data['token_string'] = data['tokens'].apply(lambda x: ' '.join(x))

    logger.info(f"Final vocab size after filtering: {len(set(token for tokens in data['tokens'] for token in tokens))}")

    logger.info('saving data')
    with open(DATA_OUTPUT_PATH, mode='w', encoding='utf-8') as f:
        data.to_csv(f, encoding='utf-8', index=False)

    logger.info('pipeline finished successfully')