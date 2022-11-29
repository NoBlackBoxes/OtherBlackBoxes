# AI : diffusion

## Install Stable Diffusion (v2.0)

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
cd ../stablediffusion
python3 setup.py develop
pip install -r requirements.txt
pip install invisible-watermark
pip install cutlass

# If using diffusers library...
pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy
```

Install Xformers (requires recent nvcc and gcc)

```bash
cd ..
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
cd ../stablediffusion
```

Download weights

```bash
mkdir weights
cd weights
wget https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt
wget https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt # Note: 4.9 Gigabytes
cd ../stablediffusion
```

Test model

```bash
# Base model
python scripts/txt2img.py --prompt "a blueprint style drawing of a school designed by buckminster fuller" --ckpt weights/512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 256 --W 256

# V model
python scripts/txt2img.py --prompt "professional photograph of a school designed by buckminster fuller" --ckpt weights/768-v-ema.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768
```
