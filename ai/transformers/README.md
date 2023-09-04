# AI : transformers

## Install

Create (and enter) Python virtual environment

```bash
mkdir tmp
cd tmp
mkdir venv
cd venv
python3 -m venv transformers
source transformers/bin/activate
```

Setup "Transformers" Environment

- Clone HuggingFace Repo

```bash
pip install git+https://github.com/huggingface/transformers.git
pip install timm matplotlib numpy
pip install opencv-python
```

# NVIDIA GPUs

# AMD GPUs
```bash
pip uninstall torch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2
```
