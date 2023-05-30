import os
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import openai
import time
from dotenv import load_dotenv
from termcolor import colored
import pyaudio
import sound

# Reload
import importlib
importlib.reload(sound)

# Load speech generation model
processor_gen = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model_gen = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Load sppech recognition processor and model
processor_rec = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model_rec = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Indicate OpenAI API Key environmental variable
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initiliaze microphone thread
microphone = sound.microphone(1600, pyaudio.paInt16, 16000, 10)
microphone.start()

# Initiliaze speaker thread
speaker = sound.speaker(1600, pyaudio.paInt16, 16000)
speaker.start()

# Initialize conversation history
conversation = [
    {"role": "system", "content": "You are a helpful chatbot named Milly."},
    {"role": "user", "content": "You are a helpful chatbot named Milly."},
    {'role': 'user', 'content': "Can you please answer my questions in a clear and entertaining way? Please keep your answers short, one ro two sentences at the most. My name is Alice and I love dogs, sushi, and Eurovision."}
]

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Three test recordings/outputs
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
        stream=False,
    )
    result =  response.choices[0].message.content

    # Voice response
    print(colored(result, "light_magenta"), end='', flush=True) # Print the response
    inputs = processor_gen(text=result, return_tensors="pt")
    speech = model_gen.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    buffer = speech.numpy()
    speaker.write(buffer)
    while speaker.playing == True:
        time.sleep(0.05)
    print("\n---\n")

    # Append to answer to conversation
    conversation.append({'role': 'assistant', 'content': f'{result}'})

# Shutdown
microphone.stop()
speaker.stop()

# FIN
