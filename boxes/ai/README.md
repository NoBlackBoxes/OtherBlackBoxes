# AI

## Setup

Create (and enter) Python virtual environment

```bash
mkdir _tmp
cd _tmp
mkdir venv
cd venv
python3 -m venv ai
source ai/bin/activate
```

Install Python libraries

```bash
pip install numpy matplotlib pillow pandas openai werkzeug python-dateutil python-dotenv nltk pyttsx3 sentencepiece
pip install torch torchvision torchaudio
```

Install espeak