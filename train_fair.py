# train_fair.py
import os
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import DataLoader, Dataset
import random

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, processor, max_len=16000*5, limit=500):
        self.root_dir = root_dir
        self.processor = processor
        self.max_len = max_len
        self.file_list = []
        self.gender_labels = []
        for speaker in os.listdir(root_dir):
            speaker_path = os.path.join(root_dir, speaker)
            if os.path.isdir(speaker_path):
                for chapter in os.listdir(speaker_path):
                    chapter_path = os.path.join(speaker_path, chapter)
                    if os.path.isdir(chapter_path):
                        for file in os.listdir(chapter_path):
                            if file.endswith('.flac'):
                                self.file_list.append(os.path.join(chapter_path, file))
                                speaker_id = int(speaker)
                                gender = 0 if speaker_id % 2 == 0 else 1  # 0: male, 1: female
                                self.gender_labels.append(gender)
        self.file_list = self.file_list[:limit]
        self.gender_labels = self.gender_labels[:limit]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0)

        # Trim/pad to max_len (now waveform length is in samples)
        if waveform.shape[0] > self.max_len:
            waveform = waveform[:self.max_len]
        else:
            pad = self.max_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # Process audio
        input_values = self.processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").input_values.squeeze()

        # Dummy transcription
        labels = self.processor(text="hello world", return_tensors="pt").input_ids.squeeze()

        gender = self.gender_labels[idx]
        return input_values, labels, gender

def train_epoch(model, processor, loader, optimizer, device, lambda_fair=0.1):
    model.train()
    total_loss = 0
    loss_female = 0.0
    loss_male = 0.0
    count_female = 0
    count_male = 0

    for input_values, labels, genders in loader:
        input_values = input_values.to(device)
        labels = labels.to(device)
        outputs = model(input_values).logits  # (batch, T_out, vocab)

        # input_lengths for CTC: number of frames after subsampling
        # For Wav2Vec2, subsampling factor is 320: T_out = ceil(input_len / 320)
        input_lengths = torch.full((input_values.size(0),), outputs.size(1), dtype=torch.long).to(device)
        # target_lengths: length of each label sequence
        target_lengths = torch.full((labels.size(0),), labels.size(1), dtype=torch.long).to(device)

        loss = nn.functional.ctc_loss(outputs.permute(1,0,2), labels, input_lengths, target_lengths, reduction='none')

        for i, gender in enumerate(genders):
            if gender == 0:
                loss_male += loss[i].item()
                count_male += 1
            else:
                loss_female += loss[i].item()
                count_female += 1

        total_loss += loss.sum().item()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    avg_loss_m = loss_male / max(count_male, 1)
    avg_loss_f = loss_female / max(count_female, 1)
    fair_term = torch.var(torch.tensor([avg_loss_m, avg_loss_f]))
    return total_loss / len(loader), fair_term

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

    data_root = '/root/ques2/LibriSpeech/train-clean-5'
    dataset = LibriSpeechDataset(data_root, processor, limit=100)
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        collate_fn=lambda x: (torch.stack([xi[0] for xi in x]),
                                              torch.stack([xi[1] for xi in x]),
                                              torch.tensor([xi[2] for xi in x])))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 3
    for epoch in range(epochs):
        loss, fair = train_epoch(model, processor, loader, optimizer, device, lambda_fair=0.1)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Fairness penalty = {fair:.4f}")

    model.save_pretrained("fair_asr_model")
    processor.save_pretrained("fair_asr_model")

if __name__ == '__main__':
    main()
