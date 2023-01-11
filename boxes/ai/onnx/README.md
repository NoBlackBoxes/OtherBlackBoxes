# AI : ONNX

## Install

Create (and enter) Python virtual environment

```bash
mkdir tmp
cd tmp
mkdir venv
cd venv
python3 -m venv onnx
source onnx/bin/activate
```

Install exporters

```bash
pip install optimum[exporters]
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git
```

Install ONNX

```bash
pip install numpy matplotlib pillow
pip install onnx
pip install onnxruntime
```

## Export models to ONNX

```bash
cd tmp
mkdir models
optimum-cli export onnx --model google/mobilenet_v2_1.0_224 models/mobilenet_v2_1.0_224/
optimum-cli export onnx --model facebook/detr-resnet-50 models/detr-resnet-50/
optimum-cli export onnx --model openai/whisper-tiny models/whisper-tiny/
optimum-cli export onnx --model stabilityai/stable-diffusion-2-1 models/stable-diffusion-2-1
```

