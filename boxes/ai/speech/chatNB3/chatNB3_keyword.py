import os
import sys
import openai
import torch
import pyttsx3
import pyaudio
import wave
import numpy as np
from python_speech_features import mfcc

# Locals libs
import libs.keyword_model as model
import libs.NB3_sound as sound

# Reimport
import importlib
importlib.reload(model)
importlib.reload(sound)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/repos/OtherBlackBoxes'
box_path = repo_path + '/boxes/ai/speech/chatNB3'
model_path = box_path + '/models/sheila.pt'

# Set OpenAI API Key (secret!!!)
openai.api_key = "<secret>"

# Initialize conversation history
conversation = [
    {"role": "system", "content": "You are small two wheeled robot shaped like a brain. Your name is NB3, which stands for no black box bot. Your task is to respond to questions about neuroscience and technology, or anything really, with a short snarky but accurate reply."},
]

# Initialize speech engine
engine = pyttsx3.init()

# Set model parameters
num_mfcc = 16
len_mfcc = 16

# Load model
keyword_model = model.custom()
keyword_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
keyword_model.to(device)

# Initiliaze microphone thread
microphone = sound.microphone(4, 1600, pyaudio.paInt16, 16000, 10)
microphone.start()

# --------------------------------------------------------------------------------
# Chat Loop
# --------------------------------------------------------------------------------
try:
    while True:

        # Wait for keyword
        while True:
            buffer = microphone.read_latest(16000)

            # Compute MFCCs
            mfccs = mfcc(buffer, 
                        samplerate=16000,
                        winlen=0.100,
                        winstep=0.064,
                        numcep=num_mfcc,
                        nfilt=num_mfcc,
                        nfft=4096,
                        preemph=0.0,
                        ceplifter=0,
                        appendEnergy=False,
                        winfunc=np.hanning)

            # Prepare neural network input
            input = torch.tensor(np.float32(mfccs.transpose()))
            input = torch.unsqueeze(torch.unsqueeze(input, 0), 0)

            # Send to GPU
            input = input.to(device)

            # Run inference
            output = keyword_model(input)

            # Extract output
            output = output.cpu().detach().numpy()
            output = np.squeeze(output)

            # If detected?
            if (output > 0.75):
                break

        # Record WAV file for input
        # ???

        # Get transcription from Whisper
        audio_file= open("speech.wav", "rb")
        transcription = openai.Audio.transcribe("whisper-1", audio_file)['text']
        conversation.append({'role': 'user', 'content': f'{transcription}'})

        # Get ChatGPT response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=conversation
        )

        # Extract and display reply
        reply = response['choices'][0]['message']['content']
        conversation.append({'role': 'assistant', 'content': f'{reply}'})

        # Speak reply
        engine.say(reply)
        engine.runAndWait()

finally:
    # Shutdown
    microphone.stop()
# FIN