import pyaudio
import sound

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

print("\n\nInput Devices\n-----------------\n")
for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print(" - Devices id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
print("\nOutput Devices\n-----------------\n")
for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
        print(" - Devices id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
p.terminate()
print("-----------------\n\n")

# Initiliaze microphone thread
# - Device 4 - internal
microphone = sound.microphone(4, 1600, pyaudio.paInt16, 16000, 10)
microphone.start()

# Initiliaze speaker thread
# - Device 4 - internal
# - Device 10 - bluetooth
speaker = sound.speaker(10, 1600, pyaudio.paInt16, 16000)
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