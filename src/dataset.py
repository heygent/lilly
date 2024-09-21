import re
import numpy as np

import torch
from torch.utils.data import Dataset


class GSM8KDataset(Dataset):
    def __init__(self, data, seq_length=128):
        self.seq_length = seq_length
        self.text = ' '.join(data['question'])
        self.text = self.text.lower()

        # Replace numbers with special tokens
        self.text = re.sub(r'\d+', lambda x: f"<num>{x.group(0)}</num>", self.text)

        # Tokenize at character level including <num> tokens
        self.tokens = re.findall(r'<num>\d+</num>|.', self.text)
        self.vocab = sorted(set(self.tokens))
        self.token2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}

        self.vocab_size = len(self.vocab)

        # Encode the entire text
        self.encoded = np.array([self.token2idx[tok] for tok in self.tokens])

    def __len__(self):
        return len(self.encoded) - self.seq_length

    def __getitem__(self, idx):
        seq = self.encoded[idx:idx + self.seq_length]
        target = self.encoded[idx + 1: idx + self.seq_length + 1]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

