# voiced_unvoiced.py
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft
import sys
import os

def real_cepstrum(frame):
    power_spec = np.abs(rfft(frame))**2
    log_spec = np.log(power_spec + 1e-10)
    ceps = irfft(log_spec)
    return ceps

def detect_voiced(audio_path, sr=16000, frame_len_ms=25, hop_len_ms=10):
    signal, sr = librosa.load(audio_path, sr=sr, mono=True)

    frame_len = int(sr * frame_len_ms / 1000)
    hop_len = int(sr * hop_len_ms / 1000)
    voiced = []  # decisions per frame

    # Loop over frames
    for start in range(0, len(signal) - frame_len + 1, hop_len):
        frame = signal[start:start+frame_len]
        ceps = real_cepstrum(frame)

        # Low quefrency cutoff (e.g., 2 ms in samples)
        low_quef_idx = int(2e-3 * sr)  # 2 ms
        if low_quef_idx >= len(ceps):
            voiced.append(0)
            continue

        high_quef = ceps[low_quef_idx:]
        max_high = np.max(high_quef)
        max_all = np.max(ceps)
        # Heuristic: if max in high quefrency > 0.1 * max of cepstrum -> voiced
        if max_high > 0.1 * max_all:
            voiced.append(1)
        else:
            voiced.append(0)

    # Expand to sample‑level: assign each sample the decision of its containing frame
    voiced_samples = np.zeros(len(signal), dtype=int)
    for i, decision in enumerate(voiced):
        start = i * hop_len
        end = start + frame_len
        if end > len(signal):
            end = len(signal)
        voiced_samples[start:end] = decision

    return signal, voiced_samples

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python voiced_unvoiced.py <audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: file {audio_file} not found.")
        sys.exit(1)

    signal, voiced = detect_voiced(audio_file)
    sr = 16000  # fixed sample rate; adjust if needed

    # Plot
    time = np.arange(len(signal)) / sr
    plt.figure(figsize=(12, 4))
    plt.plot(time, signal, label='Waveform', alpha=0.7)
    # Fill voiced regions
    plt.fill_between(time, -1, 1, where=voiced==1, alpha=0.3, color='green', label='Voiced')
    plt.ylim(-1, 1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Voiced/Unvoiced Segmentation')
    plt.tight_layout()
    plt.savefig('voiced_unvoiced.png', dpi=150)
    plt.close()
    print("Voiced/unvoiced plot saved as voiced_unvoiced.png")
