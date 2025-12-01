# clone_voice

# Imports
import os
from TTS.api import TTS

# Specify paths
base_path = "/home/kampff/Downloads/"
references_folder = f"{base_path}/references"
output_path = f"{base_path}/test_clone_output.wav"

# Specify parameters
SPEAKER_ID = "MyClonedVoice"
SAMPLE_TEXT = "Hello, this is a sample of the cloned voice."
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
#MODEL_NAME = "tts_models/en/vctk/vits" 
LANGUAGE = "en"
USE_GPU = False

# Help functuib: Get all audio reference files in a folder
def collect_reference_files(directory_path):
    valid_exts = (".wav", ".flac", ".mp3")
    files = []
    for name in sorted(os.listdir(directory_path)):
        full_path = os.path.join(directory_path, name)
        if os.path.isfile(full_path) and full_path.lower().endswith(valid_exts):
            files.append(os.path.abspath(full_path))
    return files

# --------------------------------------------------------------------------------

# Clone Voice
print("Model name:", MODEL_NAME)
print("Speaker id:", SPEAKER_ID)
print("Language:", LANGUAGE)
print("Reference directory:", output_path)

# Get references
reference_files = collect_reference_files(references_folder)
if not reference_files:
    raise RuntimeError("No reference audio files found in directory: " + references_folder)
print("Number of reference files:", len(reference_files))

# Load TTS model.
tts = TTS(MODEL_NAME, gpu=USE_GPU)

# Create / cache the speaker by synthesizing a sample once
# with both speaker_wav (reference audio) and speaker (id).
tts.tts_to_file(
    text=SAMPLE_TEXT,
    file_path=output_path,
    speaker_wav=reference_files,
    speaker=SPEAKER_ID,
    language=LANGUAGE,
)

print("Sample written to:", output_path)
print("Cloned voice cached under speaker id:", SPEAKER_ID)

