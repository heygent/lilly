import torch
import torch.nn as nn
import numpy as np
import re

class Lilly(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dim_feedforward, token2idx):
        super(Lilly, self).__init__()
        self.embedding_dim = embedding_dim
        self.token2idx = token2idx
        self.vocab_size = vocab_size

        # Custom Embedding Layer
        self.embedding = CustomEmbedding(vocab_size, embedding_dim, token2idx)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim)

        # Layer Normalization after Embedding
        self.embedding_norm = nn.LayerNorm(embedding_dim)

        # Transformer Encoder Layers with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            norm_first=True  # Apply layer normalization before sublayers
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Layer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src):
        """
        Args:
            src: Input sequence of shape (batch_size, seq_length)
        Returns:
            Output logits of shape (batch_size, seq_length, vocab_size)
        """
        src_emb = self.embedding(src) * np.sqrt(self.embedding_dim)
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.embedding_norm(src_emb)

        # Generate causal mask
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)

        output = self.transformer_encoder(
            src_emb.transpose(0, 1),  # Transformer expects (seq_length, batch_size, embedding_dim)
            mask=src_mask
        )
        output = self.output_layer(output.transpose(0, 1))
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, token2idx):
        super(CustomEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.token2idx = token2idx
        self.idx2token = {idx: tok for tok, idx in token2idx.items()}

        # Identify numerical tokens
        self.num_tokens = {tok for tok in token2idx if re.match(r'<num>\d+</num>', tok)}
        self.num_indices = [token2idx[tok] for tok in self.num_tokens]

        # Learned embeddings for non-number tokens
        self.learned_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Compute mean and std of numerical values for normalization
        num_values = []
        for tok in self.num_tokens:
            num_str = re.findall(r'\d+', tok)[0]
            n = float(num_str)
            num_values.append(n)
        self.mean = np.mean(num_values)
        self.std = np.std(num_values) + 1e-6  # Add epsilon to prevent division by zero

    def number_embedding_function(self, num_token):
        # Extract numerical value from token
        num_str = re.findall(r'\d+', num_token)[0]
        n = float(num_str)

        # Normalize n
        n = (n - self.mean) / self.std

        # Apply softsign function to map into (-1, 1)
        softsign_n = n / (1 + abs(n))

        # Generate embedding vector using polynomial embeddings with softsign
        embedding = torch.tensor([softsign_n ** (i + 1) for i in range(self.embedding_dim)], dtype=torch.float)

        return embedding

    def forward(self, x):
        batch_size, seq_length = x.size()
        embeddings = torch.zeros(batch_size, seq_length, self.embedding_dim, device=x.device)

        for i in range(batch_size):
            for j in range(seq_length):
                idx = x[i, j].item()
                token = self.idx2token[idx]
                if token in self.num_tokens:
                    embedding = self.number_embedding_function(token).to(x.device)
                    embeddings[i, j] = embedding
                else:
                    embeddings[i, j] = self.learned_embedding.weight[idx]
        return embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x
