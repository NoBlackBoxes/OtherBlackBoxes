import pyaudio
import sound
import time
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False, "cpu": True}
)
model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

text = "England have to be patient. Four days is still a long time. We have seen five-day Tests being done in two days. We know Ben Stokes manoeuvres his chess pieces like never before. In that last over he went from four slips to the short-ball tactic."

sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

# Initiliaze speaker thread
# - Device 4 - internal
# - Device 10 - bluetooth
speaker = sound.speaker(10, 2205, pyaudio.paInt16, 22050)
speaker.start()

# Output
wave = wav.numpy()
speaker.write(wave)

# Wait to stop talking
input("Press Enter to stop.")

# Cleanup
speaker.stop()

# FIN
