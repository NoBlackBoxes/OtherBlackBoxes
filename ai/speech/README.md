# AI : speech

## Install Whisper

Clone Repository

```bash
mkdir _tmp
cd _tmp
git clone https://github.com/openai/whisper
```

Create (and enter) Python virtual environment

```bash
mkdir venv
cd venv
python3 -m venv speech
source speech/bin/activate
```

Setup pytorch

```bash
cd ../whisper
python3 setup.py develop
pip install -r requirements.txt
```

Test model

```bash
whisper data/test.wav --language English
```

