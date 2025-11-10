# Dereverb
import numpy as np
import torch
import soundfile as sf
from nemo.collections.audio.models import AudioToAudioModel

# Set paths
in_path = "/home/kampff/Videos/_uploads/HTML_mono.wav"
out_path = "/home/kampff/Videos/_uploads/HTML_enhanced.wav"

# Load model
model = AudioToAudioModel.from_pretrained('nvidia/se_der_sb_16k_small').to('cpu').half()
model.eval()
model = model.float()

# Load Audio (must be 16 kHz and Mono)
audio, sr = sf.read(in_path, dtype="float32")
if sr != model.sample_rate: raise ValueError("Expected 16 kHz input")
if audio.ndim == 2: audio = audio.mean(axis=1)

# Process audio (dereverb)
chunk_size = 5 * sr  # 5 seconds
out = []
with torch.no_grad():
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        x = torch.from_numpy(chunk).to(dtype=torch.float32).view(1, 1, -1)
        l = torch.tensor([x.size(-1)], dtype=torch.int64)
        y, _ = model(input_signal=x, input_length=l)
        out.append(y.squeeze().float().numpy())
        print(len(out))
        if ((i % 12) == 0): # Save intermediate result
            print("...saved")
            sf.write(out_path, np.concatenate(out), sr)
sf.write(out_path, np.concatenate(out), sr)

# Finish
print(f"Dereverb audio saved to {out_path}")
