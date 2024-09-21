import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader

from model import Lilly

from dataset import GSM8KDataset
from datasets import load_dataset

dataset = load_dataset("gsm8k", "main")

train_data = dataset['train'] # pyright: ignore[reportIndexIssue]
gsm_dataset = GSM8KDataset(train_data)

batch_size = 32  # Smaller batch size to fit into memory

data_loader = DataLoader(gsm_dataset, batch_size=batch_size, shuffle=True)


# Hyperparameters

embedding_dim = 64
nhead = 4
num_layers = 2
dim_feedforward = 128
epochs = 5
learning_rate = 0.001

# Instantiate the model
model = Lilly(
    vocab_size=gsm_dataset.vocab_size,
    embedding_dim=embedding_dim,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    token2idx=gsm_dataset.token2idx
)

# Move model to device
device_id = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

device = torch.device(device_id)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) # pyright: ignore [reportPrivateImportUsage]

def train_model(model, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output.view(-1, gsm_dataset.vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}] completed with average loss: {epoch_loss / len(data_loader):.4f}')


