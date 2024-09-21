import torch

import numpy as np

import re
from dataset import GSM8KDataset
from datasets import load_dataset

device_id = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = torch.device(device_id)

dataset = load_dataset("gsm8k", "main")

train_data = dataset['train'] # pyright: ignore[reportIndexIssue]
gsm_dataset = GSM8KDataset(train_data)

def generate_text(model, start_text, predict_len=100):
    model.eval()
    tokens = re.findall(r'<num>\d+</num>|.', start_text.lower())
    input_indices = [gsm_dataset.token2idx.get(tok, gsm_dataset.token2idx[' ']) for tok in tokens]
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

    generated = tokens.copy()

    with torch.no_grad():
        for _ in range(predict_len):
            output = model(input_tensor)
            next_token_logits = output[0, -1]
            probabilities = torch.nn.functional.softmax(next_token_logits, dim=0).cpu().numpy()
            next_token_idx = np.random.choice(len(probabilities), p=probabilities)
            next_token = gsm_dataset.idx2token[next_token_idx]
            generated.append(next_token)
            input_indices.append(next_token_idx)
            input_tensor = torch.tensor([input_indices[-gsm_dataset.seq_length:]], dtype=torch.long).to(device)

    return ''.join(generated).replace('<num>', '').replace('</num>', '')


