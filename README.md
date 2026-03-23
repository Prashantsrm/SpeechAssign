# Question 1: Manual Cepstral Feature Extraction & Phoneme Boundary Detection

This repository contains the implementation of a hand‑crafted MFCC extraction pipeline, spectral leakage analysis, voiced/unvoiced segmentation, and phonetic boundary alignment using a pre‑trained Wav2Vec2 model.

## Requirements

- Ubuntu 22.04 or later (or any Linux distribution)
- Python 3.8+
- `pip` and `venv`

## Setup

1. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv speech_env
   source speech_env/bin/activate


2. Install dependencies

bash
pip install -r requirements.txt
Prepare audio data
The scripts expect a mono audio file (.wav or .flac) sampled at 16 kHz.
For testing, you can use a sample from the LibriSpeech dataset (already provided in data/).
If you need to create a dummy test file, run:

bash
python -c "import numpy as np, soundfile as sf; fs=16000; t=np.linspace(0,1,fs); x=np.sin(2*np.pi*440*t); sf.write('sample.wav', x, fs)"
Running the Code
All scripts accept the path to an audio file as a command‑line argument.

1. Manual MFCC Extraction
bash
python mfcc_manual.py <audio_file>
Output:

mfcc.png – a heatmap of the MFCC coefficients.

2. Spectral Leakage & SNR Analysis
bash
python leakage_snr.py <audio_file>
Output:

leakage_comparison.png – FFT magnitude plots for rectangular, Hamming, and Hann windows.

3. Voiced/Unvoiced Segmentation
bash
python voiced_unvoiced.py <audio_file>
Output:

voiced_unvoiced.png – waveform with voiced regions highlighted in green.

4. Phonetic Mapping & RMSE Calculation
bash
python phonetic_mapping.py <audio_file>
Output:

rmse_table.txt – a table containing the RMSE between the manually detected boundaries and the Wav2Vec2 forced alignment boundaries.
The first run will download the Wav2Vec2 model (approx. 1 GB).

Example (using a FLAC file from LibriSpeech)
bash
python mfcc_manual.py data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac
python leakage_snr.py data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac
python voiced_unvoiced.py data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac
python phonetic_mapping.py data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac
Results
All generated plots and the RMSE table are saved in the current working directory.
For a complete report, collect these outputs along with the textual descriptions of the methods and hyperparameters used.
