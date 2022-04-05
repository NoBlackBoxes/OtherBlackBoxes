from pydub import AudioSegment
from pydub.playback import play

song = AudioSegment.from_wav('/home/gaspard/Downloads/rocket.wav')
play(song)


print("done")