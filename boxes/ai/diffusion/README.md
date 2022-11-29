# AI : diffusion

## Install

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
python3 -m venv pytorch
source pytorch/bin/activate
```

Setup pytorch

```bash
cd ../stable_diffusion
python3 setup.py develop
pip install transformers==4.19.2 diffusers invisible-watermark omegaconf einops pytorch_lightning taming-transformers clip kornia
```

Fix VectorQuantizer (remove VectorQuantizer2 refs)