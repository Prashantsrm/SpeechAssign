# data_loader.py
import torch
import torchaudio
import random
import os
from torch.utils.data import Dataset

class LibriSpeechWithNoise(Dataset):
    def __init__(self, root_dir, noise_dir=None, noise_snr_db=10, sample_rate=16000, n_mels=80, fixed_frames=200):
        self.root_dir = root_dir
        self.noise_dir = noise_dir
        self.noise_snr_db = noise_snr_db
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fixed_frames = fixed_frames
        self.file_list = []
        self.raw_speaker_ids = []   # store raw IDs for building mapping
        # Gather all FLAC files
        for speaker in os.listdir(root_dir):
            speaker_path = os.path.join(root_dir, speaker)
            if os.path.isdir(speaker_path):
                for chapter in os.listdir(speaker_path):
                    chapter_path = os.path.join(speaker_path, chapter)
                    if os.path.isdir(chapter_path):
                        for file in os.listdir(chapter_path):
                            if file.endswith('.flac'):
                                self.file_list.append(os.path.join(chapter_path, file))
                                self.raw_speaker_ids.append(int(speaker))
        # Create mapping from raw speaker ID to consecutive index
        unique_ids = sorted(set(self.raw_speaker_ids))
        self.speaker_to_idx = {sid: i for i, sid in enumerate(unique_ids)}
        self.num_speakers = len(unique_ids)

        # Load noise files if provided
        self.noise_files = []
        if noise_dir and os.path.exists(noise_dir):
            for root, _, files in os.walk(noise_dir):
                for f in files:
                    if f.endswith('.wav'):
                        self.noise_files.append(os.path.join(root, f))
        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=400, hop_length=160
        )

    def __len__(self):
        return len(self.file_list) * 2  # clean + noisy

    def __getitem__(self, idx):
        is_noisy = idx >= len(self.file_list)
        file_path = self.file_list[idx % len(self.file_list)]
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.squeeze(0)  # mono

        # Speaker ID mapping
        raw_speaker_id = self.raw_speaker_ids[idx % len(self.file_list)]
        speaker_id = self.speaker_to_idx[raw_speaker_id]

        if is_noisy:
            # Add noise
            if self.noise_files:
                noise_path = random.choice(self.noise_files)
                noise, nsr = torchaudio.load(noise_path)
                if nsr != self.sample_rate:
                    noise = torchaudio.functional.resample(noise, nsr, self.sample_rate)
                noise = noise.squeeze(0)
                if len(noise) < len(waveform):
                    noise = torch.cat([noise, torch.zeros(len(waveform)-len(noise))])
                else:
                    noise = noise[:len(waveform)]
                # Scale to desired SNR
                signal_power = torch.mean(waveform**2)
                noise_power = torch.mean(noise**2)
                snr_linear = 10**(self.noise_snr_db / 10)
                noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
                waveform = waveform + noise_scale * noise
            else:
                noise = torch.randn_like(waveform)
                signal_power = torch.mean(waveform**2)
                noise_power = torch.mean(noise**2)
                snr_linear = 10**(self.noise_snr_db / 10)
                noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
                waveform = waveform + noise_scale * noise

        # Convert to mel‑spectrogram
        mels = self.mel_transform(waveform.unsqueeze(0)).squeeze(0)  # (n_mels, time)
        mels = mels.T  # (time, n_mels)
        time_frames = mels.size(0)

        # Crop or pad to fixed_frames
        if time_frames >= self.fixed_frames:
            start = random.randint(0, time_frames - self.fixed_frames)
            mels = mels[start:start+self.fixed_frames, :]
        else:
            pad_size = self.fixed_frames - time_frames
            mels = torch.cat([mels, torch.zeros(pad_size, self.n_mels)], dim=0)

        env_label = 1 if is_noisy else 0
        return mels, speaker_id, env_label
