# AI : speech : keyword

Keyword detection using tflite (pytorch?).

## Create Virtual Environment

```bash
mkdir -p _tmp/venv
cd _tmp/venv
python3 -m venv keyword
source keyword/bin/activate
```

## Install prerequisites

```bash
pip3 install numpy scipy torch torchsummary matplotlib pyaudio timm python_speech_features
```

## Download dataset

```bash
mkdir -p _tmp/dataset
cd _tmp/dataset
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
tar xvf speech_commands_v0.01.tar.gz
rm speech_commands_v0.01.tar.gz
```


