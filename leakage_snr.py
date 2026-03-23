# leakage_snr.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import fft, fftfreq
import librosa
import sys
import os

def analyze_window(signal, window_type, fs):
    window = get_window(window_type, len(signal))
    windowed = signal * window
    fft_vals = np.abs(fft(windowed))
    freqs = fftfreq(len(windowed), 1/fs)
    # Return positive frequencies only
    return freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2]

def main(audio_path):
    # Load a short segment (first 25 ms)
    signal, fs = librosa.load(audio_path, sr=None, mono=True)
    frame_len = int(0.025 * fs)
    frame = signal[:frame_len] if len(signal) >= frame_len else signal

    windows = ['rectangular', 'hamming', 'hann']   # corrected window name
    plt.figure(figsize=(10, 6))
    for w in windows:
        freqs, mag = analyze_window(frame, w, fs)
        # Convert to dB (avoid log(0))
        mag_db = 20 * np.log10(mag + 1e-10)
        plt.plot(freqs, mag_db, label=w.capitalize())
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.title('Spectral Leakage Comparison')
    plt.grid(True, alpha=0.3)
    plt.savefig('leakage_comparison.png', dpi=150)
    plt.close()
    print("Spectral leakage plot saved as leakage_comparison.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leakage_snr.py <audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: file {audio_file} not found.")
        sys.exit(1)
    main(audio_file)
