# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from data_loader import LibriSpeechWithNoise
from baseline import SpeakerEmbeddingBaseline
from proposed import DisentanglementModel
import configs.config as cfg

def train_baseline(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for mels, speaker_ids, _ in loader:
        mels = mels.to(device)
        speaker_ids = speaker_ids.to(device)
        emb = model(mels)
        loss = criterion(emb, speaker_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_proposed(model, loader, optimizer, criterion, device):
    model.train()
    total_recon = 0
    total_adv = 0
    # Add speaker classifier head if not present (or reinitialize if num_speakers changed)
    if not hasattr(model, 'speaker_classifier') or model.speaker_classifier.out_features != cfg.num_speakers:
        model.speaker_classifier = torch.nn.Linear(cfg.z_s, cfg.num_speakers).to(device)
    for mels, speaker_ids, env_labels in loader:
        mels = mels.to(device)
        speaker_ids = speaker_ids.to(device)
        env_labels = env_labels.to(device)
        recon, z_s, z_e = model(mels)
        # Reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(recon, mels)
        # Speaker classification loss from z_s
        cls_loss = criterion(model.speaker_classifier(z_s), speaker_ids)
        # Adversarial loss
        adv_loss = model.adversarial_loss(z_e, env_labels)
        loss = recon_loss + cfg.adv_weight * adv_loss + cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_recon += recon_loss.item()
        total_adv += adv_loss.item()
    return total_recon/len(loader), total_adv/len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_set = LibriSpeechWithNoise(
        cfg.librispeech_root, 
        cfg.noise_root, 
        cfg.noise_snr_db, 
        n_mels=cfg.n_mels,
        fixed_frames=cfg.fixed_frames
    )
    cfg.num_speakers = train_set.num_speakers   # set from dataset
    print(f"Number of speakers: {cfg.num_speakers}")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # Baseline
    print("Training baseline...")
    baseline = SpeakerEmbeddingBaseline(input_dim=cfg.n_mels).to(device)
    opt_b = optim.Adam(baseline.parameters(), lr=cfg.lr)
    crit = torch.nn.CrossEntropyLoss()
    for epoch in range(cfg.epochs):
        loss = train_baseline(baseline, train_loader, opt_b, crit, device)
        print(f"Epoch {epoch+1}: baseline loss {loss:.4f}")
    torch.save(baseline.state_dict(), os.path.join(cfg.results_dir, 'baseline.pth'))

    # Proposed
    print("Training proposed model...")
    proposed = DisentanglementModel(input_dim=cfg.n_mels, z_s=cfg.z_s, z_e=cfg.z_e, device=device).to(device)
    opt_p = optim.Adam(proposed.parameters(), lr=cfg.lr)
    for epoch in range(cfg.epochs):
        recon, adv = train_proposed(proposed, train_loader, opt_p, crit, device)
        print(f"Epoch {epoch+1}: recon {recon:.4f}, adv {adv:.4f}")
    torch.save(proposed.state_dict(), os.path.join(cfg.results_dir, 'proposed.pth'))

if __name__ == '__main__':
    main()
