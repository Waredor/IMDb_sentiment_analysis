import logging
import os
import yaml
import joblib
import pandas as pd

from logging.handlers import RotatingFileHandler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger('logistic_regression_pipeline_logger')
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s'
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = RotatingFileHandler(
    'logs/logistic_regression_pipeline_log.txt',
    maxBytes=1048576,
    backupCount=3
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


logger.info('configure paths')
PIPELINE_CONFIG_PATH = 'configs/project_config.yaml'

with open(PIPELINE_CONFIG_PATH, mode='r', encoding='utf-8') as f:
    yaml_data = yaml.safe_load(f)

TOKENIZED_DATA_PATH = yaml_data['data_path']
OUTPUT_MODEL_FILENAME = yaml_data['output_model_filename']
OUTPUT_MODEL_FILEPATH = os.path.join('models', OUTPUT_MODEL_FILENAME)


logger.info('load data')
data = pd.read_csv(TOKENIZED_DATA_PATH)
vectorizer = TfidfVectorizer()
positive_count = len(data[data['sentiment'] == 'positive'])
negative_count = len(data[data['sentiment'] == 'negative'])

logger.info(f"Positive reviews: {positive_count}, negative reviews: {negative_count}")


logger.info('creating tf-idf matrix')
tfidf_matrix = vectorizer.fit_transform(data['token_string'])
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)


logger.info('train test split')
X = tfidf_matrix
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


logger.info('model init')
model = LogisticRegression()


logger.info('model fit')
model.fit(X_train, y_train)


logger.info('model evaluate')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
positive_precision = precision_score(y_pred, y_test, pos_label='positive')
negative_precision = precision_score(y_pred, y_test, pos_label='negative')


logger.info(f"Accuracy: {accuracy}, "
            f"Positive precision: {positive_precision}, "
            f"Negative precision: {negative_precision}")


logger.info('save model')
joblib.dump(model, OUTPUT_MODEL_FILEPATH)


