from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
text_encoder_path = '/home/kampff/Dropbox/Voight-Kampff/Technology/Models/stable-diffusion-2-1/text_encoder/model.onnx'
vae_decoder_path = '/home/kampff/Dropbox/Voight-Kampff/Technology/Models/stable-diffusion-2-1/vae_decoder/model.onnx'
unet_path = '/home/kampff/Dropbox/Voight-Kampff/Technology/Models/stable-diffusion-2-1/unet/model.onnx'

# Load ONNX model
model = onnx.load(unet_path)
nodes = [n for n in model.graph.node]


# FIN