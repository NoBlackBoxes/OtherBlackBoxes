# TTS

# Imports
from TTS.api import TTS

# Specify paths
base_path = "/home/kampff/Downloads/"
references_folder = f"{base_path}/references"
output_path = f"{base_path}/clone_output.wav"

# Specify parameters
SPEAKER_ID = "MyClonedVoice"
SAMPLE_TEXT = "Hello, this is a sample of the cloned voice. Does it sound like me? at all? I am not convinced."
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
#MODEL_NAME = "tts_models/en/vctk/vits" 
LANGUAGE = "en"
USE_GPU = False

# --------------------------------------------------------------------------------

# TTS using cloned Voice
print("Model name:", MODEL_NAME)
print("Speaker id:", SPEAKER_ID)
print("Output file:", output_path)

# Load TTS model.
tts = TTS(MODEL_NAME, gpu=USE_GPU)

# Synthesize using the cached speaker id.
tts.tts_to_file(
    text=SAMPLE_TEXT,
    file_path=output_path,
    speaker=SPEAKER_ID,
    language=LANGUAGE,
)

print("Synthesis finished.")
print("Audio written to:", output_path)
