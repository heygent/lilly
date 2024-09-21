import re

import numpy as np
import torch
import torch.nn as nn


class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, token2idx):
        super(CustomEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.token2idx = token2idx
        self.idx2token = {idx: tok for tok, idx in token2idx.items()}

        # Identify numerical tokens
        self.num_tokens = {tok for tok in token2idx if re.match(r"<num>\d+</num>", tok)}
        self.num_indices = [token2idx[tok] for tok in self.num_tokens]
        self.non_num_indices = [
            idx for idx in range(vocab_size) if idx not in self.num_indices
        ]

        # Learned embeddings for non-number tokens
        self.learned_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize embeddings for non-number tokens
        nn.init.normal_(
            self.learned_embedding.weight[self.non_num_indices], mean=0.0, std=0.1
        )

        # No parameters for number embeddings
        # Number embeddings will be computed on-the-fly using a fixed function

    def number_embedding_function(self, num_token):
        # Extract numerical value from token, e.g., '<num>123</num>' -> 123
        num_str = re.findall(r"\d+", num_token)[0]
        num_value = float(num_str)
        # Normalize the numerical value
        num_value = num_value / 1000.0  # Adjust based on the expected range
        # Generate embedding vector
        embedding = torch.tensor(
            [np.sin(num_value * (i + 1)) for i in range(self.embedding_dim)],
            dtype=torch.float,
        )
        return embedding

    def forward(self, x):
        batch_size, seq_length = x.size()
        embeddings = torch.zeros(
            batch_size, seq_length, self.embedding_dim, device=x.device
        )

        for i in range(batch_size):
            for j in range(seq_length):
                idx = x[i, j].item()
                token = self.idx2token[idx]
                if token in self.num_tokens:
                    embedding = self.number_embedding_function(token)
                    embeddings[i, j] = embedding
                else:
                    embeddings[i, j] = self.learned_embedding.weight[idx]

        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-np.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embedding_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class Lilly(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, nhead, num_layers, dim_feedforward, token2idx
    ):
        super(Lilly, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = CustomEmbedding(vocab_size, embedding_dim, token2idx)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Final linear layer
        self.decoder = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.decoder.bias, 0)

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
