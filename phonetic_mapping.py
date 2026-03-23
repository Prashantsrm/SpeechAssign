# phonetic_mapping.py
import torch
import librosa
import numpy as np
import os
import sys
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from voiced_unvoiced import detect_voiced

def get_model_boundaries(audio_path, model_name="facebook/wav2vec2-base-960h"):
    """
    Returns a list of timestamps (in seconds) where the predicted token changes.
    This approximates phone boundaries.
    """
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    # Load audio at 16 kHz (the model expects 16k)
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Normalize audio (optional, but recommended)
    audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-10)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # shape (1, T, vocab_size)

    # Get frame-wise predictions (most likely token)
    pred_ids = torch.argmax(logits, dim=-1).squeeze().numpy()

    # Frame stride: model's output stride = 20 ms (16000 samples per second, 320 samples per frame)
    # For Wav2Vec2, the output length = ceil(len(audio) / 320)
    frame_duration = 0.02  # seconds per frame
    boundaries = []
    prev = None
    for i, token in enumerate(pred_ids):
        if token != prev and i > 0:
            boundaries.append(i * frame_duration)
        prev = token
    # Also add the final boundary at the end of audio
    if boundaries and boundaries[-1] < len(audio)/16000:
        boundaries.append(len(audio)/16000)
    return boundaries

def compute_rmse(boundaries_model, boundaries_ours):
    """
    Compute RMSE between two sets of boundaries (in seconds).
    Each set is a list of timestamps.
    """
    # Align by matching each of our boundaries to the nearest model boundary
    errors = []
    for b_ours in boundaries_ours:
        # Find closest model boundary
        if not boundaries_model:
            errors.append(0)
            continue
        closest = min(boundaries_model, key=lambda x: abs(x - b_ours))
        errors.append(abs(b_ours - closest))
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    return rmse

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phonetic_mapping.py <audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: file {audio_file} not found.")
        sys.exit(1)

    # Get our boundaries (voiced/unvoiced segment boundaries)
    signal, voiced = detect_voiced(audio_file)
    sr = 16000
    # Find indices where voiced changes
    boundaries_ours = []
    for i in range(1, len(voiced)):
        if voiced[i] != voiced[i-1]:
            boundaries_ours.append(i / sr)

    # Get model boundaries
    print("Loading Wav2Vec2 model (this may take a moment)...")
    boundaries_model = get_model_boundaries(audio_file)

    # Compute RMSE
    if boundaries_ours and boundaries_model:
        rmse = compute_rmse(boundaries_model, boundaries_ours)
        print(f"RMSE between our boundaries and model boundaries: {rmse:.4f} seconds")
    else:
        rmse = float('inf')
        print("No boundaries found in one of the sets.")

    # Save RMSE to a table (for report)
    with open("rmse_table.txt", "w") as f:
        f.write(f"Audio file: {audio_file}\n")
        f.write(f"Number of our boundaries: {len(boundaries_ours)}\n")
        f.write(f"Number of model boundaries: {len(boundaries_model)}\n")
        f.write(f"RMSE (seconds): {rmse:.4f}\n")
