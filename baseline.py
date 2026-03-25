# baseline.py
import torch
import torch.nn as nn

class SpeakerEmbeddingBaseline(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, embedding_dim=192, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, mels):
        out, _ = self.lstm(mels)
        out = out.mean(dim=1)
        emb = self.fc(out)
        return emb
