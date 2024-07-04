# AI : deepfakes

Creating autoencoder-based deepfakes (face swaps) from scratch (in python/pytorch)

0. Dataset generation
1. Face Extraction
2. Face alignment
3. Training (Joint-Encoder and Seperate-Decoder)
4. Inpsection
5. Swapping

## Requirements
*Note:* All commands should be run inside the "ai/deepfakes" folder

1. Create a python virtual environment

```bash
mkdir _tmp
cd _tmp
python -m venv DF
cd ..
```

2. Activate virtual environment

```bash
source _tmp/DF/bin/activate
```

3. Install required python libraries/packages

```bash
pip install numpy matplotlib torch torchvision torchsummary python-dotenv opencv-python
```

*Note*: On Windows, you may need to downgrade to numpy 1.26.4

For AMD GPUs
```bash
pip uninstall torch
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm6.0
```

4. Download model weights

```bash
cd _tmp
mkdir models
cd models

# Download detection model weights (https://github.com/1adrianb/face-alignment)
wget -O detection.pth https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
# - based on: https://arxiv.org/abs/1708.05237

# Download detection model weights (https://github.com/hhj1897/face_alignment/tree/master)
wget -O alignment.pth https://github.com/ibug-group/face_alignment/raw/master/ibug/face_alignment/fan/weights/2dfan2.pth
# - based on: https://arxiv.org/abs/1708.05237
```

## Environment

You will need a .env file in the site root directory.

```bash
BASE_PATH="/home/kampff/NoBlackBoxes/OtherBlackBoxes/ai/deepfakes"
LIBS_PATH="/home/kampff/NoBlackBoxes/OtherBlackBoxes/ai/deepfakes/libs"
```
