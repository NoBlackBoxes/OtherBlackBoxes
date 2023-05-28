import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import openai
import time
from dotenv import load_dotenv
import pyaudio
import capture

# Reload
import importlib
importlib.reload(capture)

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

# Functions
def generate_prompt(topic):
    return """You are a super friendly dog expert wiht lots of stron opinions named Milly. Can you answer questions from a girl named Alice? Her first question is {}""".format(topic)

# Wait to start talking
input("Press Enter to start talking to Milly...")

# Initiliaze capture thread
stream  = capture.stream(1600, pyaudio.paInt16, 1, 16000, 10)
stream.start()

# Wait to stop talking
input("Press Enter to stop.")

# Read sound recorded
sound = stream.read()
stream.stop()

# Extract features
inputs = processor_rec(sound, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

# Generate IDs
generated_ids = model_rec.generate(inputs=input_features, max_new_tokens=512)

# Transcribe
transcription = processor_rec.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Print question
print(transcription)

# Get answer from Milly
CHUNK = 1600
p = pyaudio.PyAudio()
stream = p.open(format = pyaudio.paFloat32,
                channels = 1,
                rate = 16000,
                output = True,
                frames_per_buffer=CHUNK)

# Prepare chat
prompt = generate_prompt(transcription)
max_response_length = 200
response = openai.ChatCompletion.create(
    # CHATPG GPT API REQUEST
    model='gpt-4',
    messages=[
        {'role': 'user', 'content': f'{prompt}'}
    ],
    max_tokens=max_response_length,
    temperature=0.75,
    stream=True,
)

start_time = start_time = time.time()
answer = ''
for event in response:     
    event_time = time.time() - start_time
    event_text = event['choices'][0]['delta']
    answer = answer + event_text.get('content', '')

    if len(answer) > 0:
        if (answer[-1]) == '.':
            print(answer, end='', flush=True) # Print the response
            inputs = processor_gen(text=answer, return_tensors="pt")
            speech = model_gen.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            buffer = speech.numpy()
            num_chunks = len(buffer) // CHUNK
            stream.write(buffer, num_chunks*CHUNK)
            answer = ''
print("\n")

# Close and terminate the stream
stream.close()
p.terminate()



#FIN
