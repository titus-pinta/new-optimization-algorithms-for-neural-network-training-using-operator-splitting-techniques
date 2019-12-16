import torch.nn as nn
import torch


class Lstm(nn.Module):
    def __init__(self, vocab_size=None, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2,
                 bidirectional=True, dropout=0.5, pad_idx=None):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(
            packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden)
