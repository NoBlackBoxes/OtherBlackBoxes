from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import os
import openai
import time
from dotenv import load_dotenv
import pyaudio

# Load speech model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


# Indicate OpenAI API Key environmental variable
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Functions
def generate_prompt(topic):
    return """You are a super friendly chatbot named Milly. Introduce youself and generate a few sentences describing {} for a curious 9 year old girl named Alice that really, really wants a dog and loves sushi. You should address Alice direclty and ask her questions about what she wants to know.""".format(topic)


# Use Whisper to have a conversation....


# Start audio output stream
CHUNK = 1600
p = pyaudio.PyAudio()
stream = p.open(format = pyaudio.paFloat32,
                channels = 1,
                rate = 16000,
                output = True,
                frames_per_buffer=CHUNK)

# Prepare chat
topic = "lightning"
prompt = generate_prompt(topic)
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
            inputs = processor(text=answer, return_tensors="pt")
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            buffer = speech.numpy()
            num_chunks = len(buffer) // CHUNK
            stream.write(buffer, num_chunks*CHUNK)
            answer = ''
print("\n")

# Close and terminate the stream
stream.close()
p.terminate()
