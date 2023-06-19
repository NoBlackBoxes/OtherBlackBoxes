import os
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import openai
import time
from dotenv import load_dotenv
from termcolor import colored
import pyaudio
import sound
import logging

# Turn off DEBUG logging
logging.getLogger().setLevel(logging.CRITICAL)

# Reload
import importlib
importlib.reload(sound)

# Load speech generation model
models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False, "cpu": True}
)
model_gen = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator_gen = task.build_generator(models, cfg)

# Load sppech recognition processor and model
processor_rec = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model_rec = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Indicate OpenAI API Key environmental variable
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initiliaze microphone thread
# - Device 4 - internal
microphone = sound.microphone(4, 1600, pyaudio.paInt16, 16000, 10)
microphone.start()

# Initiliaze speaker thread
# - Device 4 - internal
# - Device 10 - bluetooth
speaker = sound.speaker(4, 2205, pyaudio.paInt16, 22050)
speaker.start()

# Initialize conversation history
conversation = [
    {"role": "system", "content": "You are a helpful chatbot named Milly."},
    {"role": "user", "content": "You are a helpful chatbot named Milly."},
#    {'role': 'user', 'content': "Can you please answer my questions in a clear and entertaining way for a 9 year old child? Please keep your answers short, one or two sentences at the most. My name is Alice and I love dogs, sushi, and Eurovision. Also, it would be best, when appropriate, to answer in the form of a joke."}
    {'role': 'user', 'content': "Can you respond to the following transcribed snippet of a conversation that was recorded at an optics course with a witty comment or joke. If you have nothing cool to say, then just say something silly or insightful...or ask a question about similar topics."}
]

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Chat
while True:
    #
    # Input
    #

    # Wait to start talking
    input("Press <Enter> to start talking to Milly...")

    # Start recording
    microphone.reset()

    # Wait to stop talking
    input("Press <Enter> to stop talking.")
    print("\n...\n")

    # Read sound recorded
    recording = microphone.read()

    # Extract features
    inputs = processor_rec(recording, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    # Generate IDs
    generated_ids = model_rec.generate(inputs=input_features, max_new_tokens=1024)

    # Transcribe
    transcription = processor_rec.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Print question
    print("\n")
    print(colored(transcription, "light_cyan"))

    # Append to question to conversation
    conversation.append({'role': 'user', 'content': f'{transcription}'})

    #
    # Output
    #
    # Prepare chat
    max_response_length = 256
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=conversation,
        max_tokens=max_response_length,
        temperature=0.75,
        stream=True,
    )

    # Voice response
    start_time = start_time = time.time()
    partial_answer = ''
    full_answer = ''
    for event in response:     
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        partial_answer = partial_answer + event_text.get('content', '')
        if len(partial_answer) > 0:
            if ((partial_answer[-1] == '.') or (partial_answer[-1] == '!') or (partial_answer[-1] == '!')):
                sample = TTSHubInterface.get_model_input(task, partial_answer)
                wav, rate = TTSHubInterface.get_prediction(task, model_gen, generator_gen, sample)
                buffer = wav.numpy()
                speaker.write(buffer)
                time.sleep(0.50)
                while speaker.playing():
                    time.sleep(0.05)
                full_answer = full_answer + partial_answer
                partial_answer = ''
    print(colored(full_answer, "light_magenta"), end='', flush=True) # Print the response
    print("\n---\n")

    # Append to answer to conversation
    conversation.append({'role': 'assistant', 'content': f'{full_answer}'})

# Shutdown
microphone.stop()
speaker.stop()

# FIN
