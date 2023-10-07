# AI : diffusion : XL

Note: If using RunPod, then make sure to increase pos size ('Edit Pod')

## Install Stable Diffusion XL

Create (and enter) Python virtual environment

```bash
mkdir _tmp
cd _tmp
python3 -m venv OBB
source OBB/bin/activate
```

Setup "Diffusion" Environment

```bash
pip install diffusers transformers accelerate safetensors omegaconf pillow
```

## Run simple example

```python
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForText2Image
from PIL import Image
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "a hyper-realistic photograph of a completely imaginary thing."
image = pipeline(prompt=prompt).images[0]
image.save('test.png')
#FIN
```