import numpy as np
import torch

from torch.utils.data import Dataset


def tokens_to_embeddings(tokens, token_embeddings, max_seq_length, vector_size) -> np.ndarray:
    embeddings = []
    for token in tokens[:max_seq_length]:
        if token in token_embeddings:
            embeddings.append(token_embeddings[token])
        else:
            embeddings.append(np.zeros(vector_size))

    while len(embeddings) < max_seq_length:
        embeddings.append(np.zeros(vector_size))

    return np.array(embeddings)


class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 0)) for token in self.texts[idx]]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)