# Audio : Bluetooth

Using bluetooth audio on Linux

## Setup
- Install bluetoothctl (bluez) to pair and connect, ensire the service is enabled and reboot!
- Install bluez-alsa to create a virtual ALSA PCM device for use with Python's pyaudio (and other ALSA libraries)
  - Config various /etc/alsa/config.d/... (add MAC address for default device) and reboot (perhaps not necessary)