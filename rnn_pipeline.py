import logging
import os
import ast
import pickle
import yaml
import pandas as pd
import numpy as np
import torch

from logging.handlers import RotatingFileHandler
from utils.rnn_dataset import TextDataset
from utils.rnn_model import RNNModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    def collate_fn(batch):
        global word_to_idx
        input_texts, input_labels = zip(*batch)
        output_texts = pad_sequence(input_texts, batch_first=True, padding_value=word_to_idx['<PAD>'])
        return output_texts, torch.tensor(input_labels, dtype=torch.long)


    def load_glove_embeddings(glove_path, word_to_idx_dict, vector_size):
        embeddings = np.random.normal(0, 0.1, (len(word_to_idx_dict) + 2, vector_size))  # +2 для <PAD> и <UNK>
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                if word in word_to_idx_dict:
                    embeddings[word_to_idx_dict[word]] = [float(x) for x in values[1:]]
        # Добавляем эмбеддинги для <PAD> и <UNK>
        pad_idx = word_to_idx_dict['<PAD>']
        unk_idx = word_to_idx_dict['<UNK>']
        embeddings[pad_idx] = np.zeros(vector_size)
        embeddings[unk_idx] = np.random.normal(0, 0.1, vector_size)
        return embeddings


    logger = logging.getLogger('rnn_pipeline_logger')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)


    formatter = logging.Formatter(
        '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s'
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        'logs/rnn_pipeline_log.txt',
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
    GLOVE_EMBEDDINGS_DICT_PATH = yaml_data['embeddings_dir']
    PICKLE_EMBEDDINGS_DICT_PATH = os.path.join('data', 'token_embeddings.pkl')
    OUTPUT_MODEL_FILENAME = yaml_data['output_model_filename']
    NUM_EPOCHS = yaml_data['num_epochs']
    BATCH_SIZE = yaml_data['batch_size']
    LEARNING_RATE = yaml_data['lr0']
    OUTPUT_MODEL_FILEPATH = os.path.join('models', OUTPUT_MODEL_FILENAME)

    VECTOR_SIZE = 100


    with open(PICKLE_EMBEDDINGS_DICT_PATH, 'rb') as f:
        token_embeddings = pickle.load(f)

    word_to_idx = {word: idx for idx, word in enumerate(token_embeddings.keys())}
    # Добавляем <PAD> и <UNK> в word_to_idx
    word_to_idx['<PAD>'] = len(word_to_idx)
    word_to_idx['<UNK>'] = len(word_to_idx)

    token_embeddings = load_glove_embeddings(GLOVE_EMBEDDINGS_DICT_PATH, word_to_idx, VECTOR_SIZE)

    logger.info(
        f"Vocab size: {len(word_to_idx)}, GloVe coverage: {np.sum(token_embeddings.any(axis=1)) / len(word_to_idx):.4f}")

    logger.info('configure data')
    label_encoder = LabelEncoder()

    data = pd.read_csv(TOKENIZED_DATA_PATH, encoding='utf-8')
    data['tokens'] = data['tokens'].apply(ast.literal_eval)
    max_seq_length = 200


    data['label_encoded'] = label_encoder.fit_transform(data['sentiment'])


    logger.info('data train test split')
    X = data['tokens'].values
    y = data['label_encoded'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    logger.info('create dataloaders')
    embedding_matrix = torch.tensor(token_embeddings, dtype=torch.float)


    train_dataset = TextDataset(X_train, y_train, word_to_idx=word_to_idx)
    test_dataset = TextDataset(X_test, y_test, word_to_idx=word_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    logger.info(
        f"Vocab size: {len(word_to_idx)}, GloVe coverage: {np.sum(token_embeddings.any(axis=1)) / len(word_to_idx):.4f}")


    logger.info('model init')
    rnn_model = RNNModel(
        embedding_matrix=embedding_matrix,
        hidden_size=128,
        num_layers=2,
        dropout=0.4,
        word_to_idx=word_to_idx
    )


    logger.info('model fit')
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"using device: {device}")
    rnn_model.to(device)

    for epoch in range(NUM_EPOCHS):
        train_sumloss = 0
        num_batches = 0
        rnn_model.train()
        for batch_idx, (texts, labels) in enumerate(train_loader):
            num_batches += 1
            optimizer.zero_grad()
            texts = texts.to(device)
            labels = labels.to(device)
            output = rnn_model(texts)
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_sumloss += loss.item()
            #logger.info(f"batch: {batch_idx + 1}, batch_loss: {loss.item()}")

        train_sumloss /= num_batches

        num_batches = 0
        test_sumloss = 0
        all_preds = []
        all_labels = []
        rnn_model.eval()
        with torch.no_grad():
            for batch_idx, (texts, labels) in enumerate (test_loader):
                num_batches += 1
                texts = texts.to(device)
                labels = labels.to(device)
                output = rnn_model(texts)
                loss = criterion(output, labels)
                test_sumloss += loss.item()
                preds = torch.argmax(output, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        test_sumloss /= num_batches
        accuracy = accuracy_score(all_labels, all_preds)

        logger.info(f"epoch: {epoch + 1}, train loss: {train_sumloss}, "
                    f"test loss: {test_sumloss}, "
                    f"test accuracy: {accuracy}")

        scheduler.step(test_sumloss)


    torch.save(rnn_model.state_dict(), OUTPUT_MODEL_FILEPATH)
    logger.info(f"Model saved to {OUTPUT_MODEL_FILEPATH}")