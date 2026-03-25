# configs/config.py
import os

# Paths
librispeech_root = '/root/ques2/LibriSpeech/train-clean-5'
noise_root = None                     # set to path to MUSAN if you have it
results_dir = './results'

# Data
noise_snr_db = 10
n_mels = 80
fixed_frames = 200                    # 2 seconds at 16kHz, hop=160
batch_size = 32

# Training
epochs = 20
lr = 1e-3
adv_weight = 0.1

# Model
z_s = 64
z_e = 32

os.makedirs(results_dir, exist_ok=True)# configs/config.py
import os

# Paths
librispeech_root = '/root/ques2/LibriSpeech/train-clean-5'
noise_root = None                     # set to path to MUSAN if you have it
results_dir = './results'

# Data
noise_snr_db = 10
n_mels = 80
fixed_frames = 200                    # 2 seconds at 16kHz, hop=160
batch_size = 32

# Training
epochs = 20
lr = 1e-3
adv_weight = 0.1

# Model
z_s = 64
z_e = 32

os.makedirs(results_dir, exist_ok=True)
