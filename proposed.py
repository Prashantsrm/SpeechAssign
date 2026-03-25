# proposed.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden=256, z_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden*2, z_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out)

class EnvironmentEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden=128, z_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden*2, z_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out)

class Decoder(nn.Module):
    def __init__(self, z_dim_speaker=64, z_dim_env=32, output_dim=80, hidden=256):
        super().__init__()
        self.fc = nn.Linear(z_dim_speaker + z_dim_env, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden*2, output_dim)

    def forward(self, z_s, z_e, seq_len):
        z = torch.cat([z_s, z_e], dim=1)
        z = self.fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(z)
        out = self.out(out)
        return out

class DisentanglementModel(nn.Module):
    def __init__(self, input_dim=80, z_s=64, z_e=32, device='cpu'):
        super().__init__()
        self.speaker_enc = SpeakerEncoder(input_dim, z_dim=z_s)
        self.env_enc = EnvironmentEncoder(input_dim, z_dim=z_e)
        self.decoder = Decoder(z_s, z_e, input_dim)
        self.discriminator = nn.Sequential(
            nn.Linear(z_e, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.device = device

    def forward(self, mels):
        z_s = self.speaker_enc(mels)
        z_e = self.env_enc(mels)
        recon = self.decoder(z_s, z_e, mels.size(1))
        return recon, z_s, z_e

    def adversarial_loss(self, z_e, env_labels):
        pred = self.discriminator(z_e)
        return F.cross_entropy(pred, env_labels)
