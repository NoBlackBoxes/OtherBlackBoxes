import pyaudio
import sound

# Initiliaze microphone thread
microphone = sound.microphone(1600, pyaudio.paInt16, 16000, 10)
microphone.start()

# Initiliaze speaker thread
speaker = sound.speaker(1600, pyaudio.paInt16, 16000)
speaker.start()

# Three test recordings/outputs
for i in range(3):
    # Wait to start talking
    input("Press Enter to start talking to Milly...")

    # Start recording
    microphone.reset()

    # Wait to stop talking
    input("Press Enter to stop.")

    # Read sound recorded
    recording = microphone.read()

    # Report
    print(len(recording))

    # Output
    speaker.write(recording)

# Shutdown
microphone.stop()
speaker.stop()

# FIN