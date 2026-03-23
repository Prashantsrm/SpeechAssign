# mfcc_manual.py
import numpy as np
import librosa
from scipy.signal import get_window
from scipy.fft import rfft
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import sys
import os

def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def framing(signal, sr, frame_len_ms=25, hop_len_ms=10):
    frame_len = int(sr * frame_len_ms / 1000)
    hop_len = int(sr * hop_len_ms / 1000)
    frames = []
    for start in range(0, len(signal) - frame_len + 1, hop_len):
        frames.append(signal[start:start+frame_len])
    return np.array(frames)

def mfcc(audio_path, sr=16000, n_mfcc=13, n_mels=40, alpha=0.97):
    # Load audio (librosa handles FLAC)
    signal, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Pre‑emphasis
    signal = pre_emphasis(signal, alpha)

    # Framing
    frames = framing(signal, sr)

    # Window
    window = get_window('hamming', frames.shape[1])

    # FFT power spectrum
    power_spec = np.abs(rfft(frames * window, axis=1))**2
    # Frequency bins for later use (not strictly needed for MFCC)
    # freqs = np.fft.rfftfreq(frames.shape[1], 1/sr)

    # Mel filterbank (use librosa's filterbank to avoid re‑implementing)
    # This is allowed because we are not using high‑level MFCC extraction.
    from librosa.filters import mel
    mel_fb = mel(sr=sr, n_fft=len(window), n_mels=n_mels, fmin=0, fmax=sr//2)

    mel_energy = np.dot(power_spec, mel_fb.T)

    # Log compression
    log_mel = np.log(mel_energy + 1e-10)

    # DCT
    mfcc = dct(log_mel, axis=1, type=2, norm='ortho')[:, :n_mfcc]

    return mfcc

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mfcc_manual.py <audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: file {audio_file} not found.")
        sys.exit(1)

    mfcc_coeffs = mfcc(audio_file)
    print(f"MFCC shape: {mfcc_coeffs.shape}")

    # Optional: plot the MFCCs
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc_coeffs.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='MFCC coefficient')
    plt.xlabel('Frame index')
    plt.ylabel('MFCC coefficient index')
    plt.title('Manual MFCCs')
    plt.savefig('mfcc.png')
    plt.close()
    print("MFCC plot saved as mfcc.png")
