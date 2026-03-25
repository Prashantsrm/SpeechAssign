# pp_demo.py
import os
import torch
import torchaudio
from privacymodule import PrivacyPreservingModule

def main():
    os.makedirs('examples', exist_ok=True)

    # Pick one LibriSpeech file (adjust path to an existing one)
    audio_path = '/root/ques2/LibriSpeech/train-clean-5/1088/134315/1088-134315-0000.flac'
    if not os.path.exists(audio_path):
        print("LibriSpeech file not found. Please adjust the path.")
        return

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    waveform = waveform.squeeze(0)  # mono

    # Save original
    torchaudio.save('examples/original.wav', waveform.unsqueeze(0), sr)

    # Male → Female (raise pitch)
    module = PrivacyPreservingModule(sample_rate=sr, shift_semitones=5)
    output_mf = module(waveform.unsqueeze(0)).squeeze()
    torchaudio.save('examples/male_to_female.wav', output_mf.unsqueeze(0), sr)

    # Female → Male (lower pitch)
    module = PrivacyPreservingModule(sample_rate=sr, shift_semitones=-5)
    output_fm = module(waveform.unsqueeze(0)).squeeze()
    torchaudio.save('examples/female_to_male.wav', output_fm.unsqueeze(0), sr)

    print("Transformed audio saved to examples/")

if __name__ == '__main__':
    main()
