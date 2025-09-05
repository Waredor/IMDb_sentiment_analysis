import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(
            self, embedding_matrix, hidden_size, num_layers, word_to_idx, dropout=0
    ) -> None:
        super(RNNModel, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=word_to_idx['<PAD>']
        )
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_layers,
            bidirectional=True, dropout=dropout, batch_first=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, _ = self.lstm(embedded)
        # Механизм внимания
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        output = torch.sum(lstm_output * attn_weights, dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        return output