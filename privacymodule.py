# privacymodule.py
import torch
import torchaudio

class PrivacyPreservingModule(torch.nn.Module):
    def __init__(self, sample_rate=16000, shift_semitones=5):
        super().__init__()
        self.sample_rate = sample_rate
        self.shift_semitones = shift_semitones
        self.resample_factor = 2 ** (shift_semitones / 12.0)

    def forward(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        batch, samples = waveform.shape
        resampled = torchaudio.functional.resample(
            waveform, self.sample_rate, int(self.sample_rate / self.resample_factor)
        )
        resampled = torchaudio.functional.resample(
            resampled, int(self.sample_rate / self.resample_factor), self.sample_rate
        )
        if resampled.shape[-1] > samples:
            resampled = resampled[:, :samples]
        else:
            pad = samples - resampled.shape[-1]
            resampled = torch.nn.functional.pad(resampled, (0, pad))
        return resampled.view(waveform.shape)
