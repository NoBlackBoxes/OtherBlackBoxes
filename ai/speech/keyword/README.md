# AI : speech : keyword

Keyword detection using tflite (pytorch?).

## Install prerequisites

```bash
pip3 install numpy scipy torch torchsummary matplotlib pyaudio timm ptflops
```

- For model conversion
```bash
pip install tensorflow ai_edge_torch # currently requires Python <=3.12

# required for some reason
sudo execstack -c  _tmp/AI.311/lib/python3.11/site-packages/ai_edge_liter
t/_pywrap_tensorflow_interpreter_wrapper.so 
```

## Download dataset

```bash
mkdir -p _tmp/dataset
cd _tmp/dataset
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar xvf speech_commands_v0.02.tar.gz
rm speech_commands_v0.02.tar.gz
```
