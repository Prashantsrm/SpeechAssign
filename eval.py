# eval.py
import torch
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from data_loader import LibriSpeechWithNoise
from baseline import SpeakerEmbeddingBaseline
from proposed import DisentanglementModel
import configs.config as cfg

def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def evaluate(model, test_loader, device, model_type='baseline'):
    model.eval()
    embeddings = []
    speaker_ids = []
    with torch.no_grad():
        for mels, spk_id, _ in test_loader:
            mels = mels.to(device)
            if model_type == 'baseline':
                emb = model(mels)
            else:
                emb = model.speaker_enc(mels)
            embeddings.append(emb.cpu())
            speaker_ids.append(spk_id)
    embeddings = torch.cat(embeddings, dim=0).numpy()
    speaker_ids = torch.cat(speaker_ids, dim=0).numpy()

    # Cosine similarity matrix
    scores = np.dot(embeddings, embeddings.T)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = scores / np.dot(norms, norms.T)

    # Positive pairs: same speaker, different utterances
    pos_mask = speaker_ids[:, None] == speaker_ids[None, :]
    np.fill_diagonal(pos_mask, False)
    triu_indices = np.triu_indices_from(pos_mask, k=1)
    pos_labels = pos_mask[triu_indices]
    scores_triu = scores[triu_indices]
    eer = compute_eer(scores_triu, pos_labels.astype(int))
    return eer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = LibriSpeechWithNoise(
        cfg.librispeech_root, 
        cfg.noise_root, 
        cfg.noise_snr_db, 
        n_mels=cfg.n_mels,
        fixed_frames=cfg.fixed_frames
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    # Baseline
    baseline = SpeakerEmbeddingBaseline(input_dim=cfg.n_mels).to(device)
    baseline.load_state_dict(torch.load('results/baseline.pth', map_location=device))
    baseline_eer = evaluate(baseline, test_loader, device, 'baseline')
    print(f"Baseline EER: {baseline_eer:.4f}")

    # Proposed
    proposed = DisentanglementModel(input_dim=cfg.n_mels, z_s=cfg.z_s, z_e=cfg.z_e, device=device).to(device)
    # Load state dict, ignoring missing keys (like speaker_classifier)
    state_dict = torch.load('results/proposed.pth', map_location=device)
    proposed.load_state_dict(state_dict, strict=False)
    proposed_eer = evaluate(proposed, test_loader, device, 'proposed')
    print(f"Proposed EER: {proposed_eer:.4f}")

    with open('results/eer_table.txt', 'w') as f:
        f.write(f"Baseline EER: {baseline_eer:.4f}\n")
        f.write(f"Proposed EER: {proposed_eer:.4f}\n")

if __name__ == '__main__':
    main()
