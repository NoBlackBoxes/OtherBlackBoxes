import sys
import openai
import pyttsx3
import pyaudio
import wave
import numpy as np
import serial
import curses
import time

# Set OpenAI API Key (secret!!!)
openai.api_key = "<secret>"

# Initialize conversation history
conversation = [
    {"role": "system", "content": "You are small two wheeled robot. Your name is NB3, which stands for no black box bot. \
     Your task is to respond to requests for you to move in a particular way with a sensible, funny, somwhat snarky text reply and a sequence of movements. \
     The movement commands should follow immediately after a '##' at the end of your text reply. There should be a final '##' at the end of the commands. \
     They should have the following format: \"<some text reply you produce>##f200 l300 r100 b75##\". \
     The commands must consist of single letters (f,b,l,r) followed by a number. f is forward, b is backward, l is left turn, r is right turn, and the numbers \
     indicate how long the robot should perform the movement for in milliseconds. So, for the previous example, the robot would move forward for 200 ms, make a \
     left turn for 300 ms, a right turn for 100 ms, and go backward for 100 ms."},
]

# Configure serial port
ser = serial.Serial()
ser.baudrate = 19200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()
time.sleep(1.50) # Wait for connection before sending any data

# Robot initial state (waiting and stopped)
ser.write(b'x')
time.sleep(0.05)
ser.write(b'w')
time.sleep(0.05)

# Initialize speech engine
engine = pyttsx3.init()

# Set sound recording format
CHUNK = 1600                # Buffer size
FORMAT = pyaudio.paInt16    # Data type
CHANNELS = 1                # Number of channels
RATE = 16000                # Sample rate (Hz)
MAX_DURATION = 5            # Max recording duration
WAVE_OUTPUT_FILENAME = "speech.wav"

# Get pyaudio object
pya = pyaudio.PyAudio()

# Open audio stream (from default device)
stream = pya.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            start=False,
            frames_per_buffer=CHUNK)

# Setup the curses screen window
screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.nodelay(True)
 
# --------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------

# Function to record speech snippets to a WAV file
def record_speech(stream):

    # Prepare a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)

    # Start streaming audio
    stream.start_stream()

    # Append frames of data until key (spacebar) is pressed
    frames = []
    for i in range(0, int(RATE / CHUNK * MAX_DURATION)):
        # Read raw data and append
        raw_data = stream.read(CHUNK)
        frames.append(raw_data)
    
        # Check for key press ('z')
        char = screen.getch()
        if char == ord('z'):
            break

    # Stop stream
    stream.stop_stream()

    # Write to WAV file
    wf.writeframes(b''.join(frames))
    
    # Close WAV file
    wf.close()

    return
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Chat Loop
# --------------------------------------------------------------------------------
try:
    while True:

        # Wait to start talking
        screen.addstr(0, 0, "Press 'z' to talk to your NB3 ('q' to quit):")
        screen.clrtoeol()
        while True:
            char = screen.getch()
            if char == ord('q'):
                sys.exit()
            elif char == ord('z'):
                break

        # Indicate hearing (stop moving and blink)
        ser.write(b'x')
        time.sleep(0.05)
        ser.write(b'h')
        time.sleep(0.05)

        # Start recording
        screen.addstr("...press 'z' again to stop speaking.", curses.A_UNDERLINE)
        record_speech(stream)
        screen.erase()        

        # Indicate done hearing (twitch and wait)
        ser.write(b'l')
        time.sleep(0.15)
        ser.write(b'x')
        time.sleep(0.05)
        ser.write(b'r')
        time.sleep(0.15)
        ser.write(b'x')
        time.sleep(0.05)
        ser.write(b'w')
        time.sleep(0.05)

        # Get transcription from Whisper
        audio_file= open("speech.wav", "rb")
        transcription = openai.Audio.transcribe("whisper-1", audio_file)['text']
        conversation.append({'role': 'user', 'content': f'{transcription}'})
        screen.addstr(4, 0, "You: {0}\n".format(transcription), curses.A_STANDOUT)
        screen.addstr(6, 0, " . . . ", curses.A_NORMAL)
        screen.refresh()

        # Get ChatGPT response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=conversation
        )

        # Extract and display reply
        reply = response['choices'][0]['message']['content']
        conversation.append({'role': 'assistant', 'content': f'{reply}'})

        # Split message from commands
        split_reply = reply.split('##')
        message = split_reply[0]
        if len(split_reply) > 1:
            command_string = split_reply[1]
        else:
            command_string = ""

        # Indicate speaking (stop moving and blink)
        ser.write(b'x')
        time.sleep(0.05)
        ser.write(b's')
        time.sleep(0.05)

        # Speak message
        engine.say(message)
        engine.runAndWait()
        screen.addstr(8, 0, "NB3: {0}\n".format(message), curses.A_NORMAL)
        screen.addstr(12, 0, "- commands: {0}\n".format(command_string), curses.A_STANDOUT)
        screen.refresh()

        # Indicate done speaking (stop moving and blink)
        ser.write(b'x')
        time.sleep(0.05)
        ser.write(b'w')
        time.sleep(0.05)

        # Execute commands
        commands = command_string.split(' ')
        if(len(commands) > 1):
            for c in commands:
                dir = c[0].encode('utf-8')
                dur = int(c[1:])
                dur_f = dur / 1000.0
                ser.write(dir)
                time.sleep(dur_f)

        # Stop
        ser.write(b'x')
        time.sleep(0.05)

finally:
    # shut down
    stream.close()
    pya.terminate()
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()
    ser.close()
# FIN