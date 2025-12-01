# AI : speech : clone
Clone voice from example, generate new audio from script

## Extract Audio from Video and Convert to 16 kHz, Mono, S16, WAV
```bash
ffmpeg -y -i video.mkv -ac 1 -ar 16000 -sample_fmt s16 -af "loudnorm" audio_16k.wav
```

## Convert Audio (MP3, WAV, etc.) to 16 kHz, Mono, S16, WAV
```bash
ffmpeg -y -i audio.mp3 -ac 1 -ar 16000 -sample_fmt s16 -af "loudnorm" audio_16k.wav
```

## Install requirements
- Create and Activate a relevant virtual environment, then...

```bash
pip install coqui-tts
```

