# evaluation_scripts/dnsmos.py
import torch
import torchaudio
import numpy as np
import os

def compute_snr(original, transformed):
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - transformed) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def evaluate_audio_pairs(original_dir='examples', transformed_dir='examples'):
    original_path = os.path.join(original_dir, 'original.wav')
    mf_path = os.path.join(transformed_dir, 'male_to_female.wav')
    fm_path = os.path.join(transformed_dir, 'female_to_male.wav')

    if not os.path.exists(original_path):
        print("No original audio found. Run pp_demo.py first.")
        return

    orig, sr = torchaudio.load(original_path)
    mf, _ = torchaudio.load(mf_path)
    fm, _ = torchaudio.load(fm_path)

    min_len = min(orig.shape[1], mf.shape[1], fm.shape[1])
    orig = orig[:, :min_len]
    mf = mf[:, :min_len]
    fm = fm[:, :min_len]

    snr_mf = compute_snr(orig, mf)
    snr_fm = compute_snr(orig, fm)

    print(f"SNR (male->female): {snr_mf:.2f} dB")
    print(f"SNR (female->male): {snr_fm:.2f} dB")

if __name__ == '__main__':
    evaluate_audio_pairs()
