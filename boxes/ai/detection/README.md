# AI : detection

## Install DETR

Clone Repository

```bash
mkdir tmp
cd tmp
git clone https://github.com/Stability-AI/stablediffusion
```

Create (and enter) Python virtual environment

```bash
mkdir venv
cd venv
python3 -m venv detection
source detection/bin/activate
```

Setup "Detection" Environment

```bash
pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy timm matplotlib
```
