#!/usr/bin/env python3

# %%
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url

import numpy as np
import os
from tqdm import tqdm

SEQLEN = 100
BATCH_SIZE = 64

data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'    # noqa: E501
data_path = 'data/shakespeare.txt'

if not os.path.exists(data_path):
    download_url(data_url, *os.path.split(data_path))


# %%
class TextDataset(Dataset):
    def __init__(self, text_file: str):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        self.__vocab = sorted(set(text))

        all_ids = self.__ids_from_chars(text)
        seqs = torch.split(all_ids, SEQLEN + 1)

        self.__seqs = [(seq[:-1], seq[1:]) for seq in seqs]

    def vocab_size(self) -> int:
        return len(self.__vocab)

    def __ids_from_chars(self, chars: str) -> torch.Tensor:
        return torch.from_numpy(np.asarray([self.__vocab.index(ch)
                                            for ch in chars]
                                           )).long()

    def __chars_from_ids(self, ids: list[int]) -> str:
        return ''.join([self.__vocab[id] for id in ids])

    def __len__(self) -> int:
        return len(self.__seqs) - 1

    def __getitem__(self, idx: int):
        return self.__seqs[idx]


# %%
class RNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.__embedding = nn.Embedding(vocab_size, embedding_dim)
        self.__rnn = nn.LSTM(input_size=embedding_dim, hidden_size=1024)
        self.__linear = nn.Linear(1024, vocab_size)
        # self.__softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__embedding(x)
        x, _ = self.__rnn(x)
        x = self.__linear(x)
        return x


# %%
dataset = TextDataset(data_path)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              pin_memory=True)
model = RNN(dataset.vocab_size(), 256)
model.cuda()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# %%
def train(epoch: int):
    for current_epoch in range(epoch):
        running_loss = []
        for inputs, labels in tqdm(train_dataloader):
            inputs: torch.Tensor = inputs.cuda(non_blocking=True)
            labels: torch.Tensor = inputs.cuda()
            outputs: torch.Tensor = model(inputs)

            optimizer.zero_grad()
            outputs = outputs.flatten(end_dim=1)
            labels = labels.flatten()

            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
        print("[%d] loss: %.3f" % (current_epoch, np.mean(running_loss)))


# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    train(8)


if __name__ == '__main__':
    main()
