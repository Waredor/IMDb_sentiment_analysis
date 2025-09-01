import os
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import nltk
import yaml

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize_text_on_words(text: str) -> list:
    global stop_words
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens


def get_project_root(start_path):
    current = start_path
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Project root was not found")


init_path = os.path.abspath(__file__)

PROJECT_ROOT_PATH = get_project_root(init_path)
LOGGER_PATH = os.path.join(PROJECT_ROOT_PATH, 'logs', 'tokenize_dataframe_log.txt')


logger = logging.getLogger('tokenize-dataframe_logger')
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s'
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = RotatingFileHandler(
    LOGGER_PATH,
    maxBytes=1048576,
    backupCount=3
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


logger.info('configuring paths')
PIPELINE_CONFIG_PATH = os.path.join(PROJECT_ROOT_PATH, 'configs', 'project_config.yaml')

with open(PIPELINE_CONFIG_PATH, mode='r', encoding='utf-8') as f:
    yaml_data = yaml.safe_load(f)

DATA_PATH = yaml_data['raw_data_path']
DATA_OUTPUT_PATH = os.path.join(PROJECT_ROOT_PATH, 'data', 'IMDB Dataset_tokenized.csv')

vectorizer = TfidfVectorizer()


logger.info('creating dataframe')

data = pd.read_csv(DATA_PATH, encoding='utf-8')
stop_words = set(stopwords.words('english'))


logger.info('tokenize text data')

data['tokens'] = data['review'].apply(tokenize_text_on_words)
data['token_string'] = data['tokens'].apply(lambda x: ' '.join(x))


logger.info('creating tf-idf matrix')

tfidf_matrix = vectorizer.fit_transform(data['token_string'])
feature_names = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df['sentiment'] = data['sentiment']


logger.info('saving data')

with open(DATA_OUTPUT_PATH, mode='w', encoding='utf-8') as f:
    data.to_csv(f, encoding='utf-8')

logger.info('pipeline finished successfully')