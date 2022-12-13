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
python3 -m venv diffusion
source diffusion/bin/activate
```

Setup "Diffusion" Environment

```bash
cd ../stablediffusion
python3 setup.py develop
pip install -r requirements.txt
pip install invisible-watermark
pip install cutlass
# pip uninstall torch
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

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
wget https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt
# Note: 4.9 Gigabytes
cd ..
```

Test model

```bash
# Base model
python scripts/txt2img.py --prompt "a blueprint style drawing of a school designed by buckminster fuller" --ckpt weights/512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 256 --W 256

# V model
python scripts/txt2img.py --prompt "professional photograph of a school designed by buckminster fuller" --ckpt weights/768-v-ema.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768
```

## ONNX

Convert stable diffusion to ONNX format

```bash
pip install onnx
pip install onnxruntime
pip install onnxruntime-openvino # Intel
cd tmp
git clone https://github.com/huggingface/diffusers
mkdir onnx
python diffusers/scripts/convert_stable_diffusion_checkpoint_to_onnx.py \
  --model_path="stabilityai/stable-diffusion-2" \
  --output_path="onnx/stablediffusion_onnx"
```


Build OpenVINO

```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make --jobs=$(nproc --all)
```