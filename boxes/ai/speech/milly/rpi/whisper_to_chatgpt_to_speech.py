import openai
import pyttsx3
import pyaudio
import wave
import numpy as np
import sound

# Set OpenAI API Key (secret!!!)
openai.api_key = "---secret---"
system_prompt = "You are small two wheeled robot shaped like a brain named NB3. Your task is to listen to snippets of audio from a neuroscience course and respond with witty comments. Only produce short one sentence replies."

# Initialize speech engine
engine = pyttsx3.init()

# Initiliaze microphone thread
# - Device 4 - internal
microphone = sound.microphone(1, 2205, pyaudio.paInt16, 22050, 10)
microphone.start()

# Loop
while True:

    # Wait to start talking
    input("Press <Enter> to start talking to NB3...")

    # Start recording
    microphone.reset()

    # Wait to stop talking
    input("Press <Enter> to stop talking.")
    print("\n...\n")

    # Read sound recorded
    recording = (microphone.read() * 32000).astype(np.int16)

    # Save a wav file
    wf = wave.open("test.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(22050)
    wf.writeframes(b''.join(recording))
    wf.close()

    # Get transcript from Whisper
    audio_file= open("test.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)['text']

    # Get ChatGPT response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcript
            }
        ]
    )

    # Extract and diaply reply
    reply = response['choices'][0]['message']['content']
    print(reply)

    # Say reply
    engine.say(reply)
    engine.runAndWait()

# Shutdown
microphone.stop()
