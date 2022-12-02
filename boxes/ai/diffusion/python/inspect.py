import torch
from omegaconf import OmegaConf # hmmmm
from tmp.stablediffusion.ldm.util import instantiate_from_config
from tmp.stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
from tmp.stablediffusion.ldm.models.diffusion.plms import PLMSSampler
from tmp.stablediffusion.ldm.models.diffusion.dpm_solver import DPMSolverSampler

# Paths
config_path = "tmp/stablediffusion/configs/stable-diffusion/v2-inference-v.yaml"
checkpoint_path = "tmp/stablediffusion/weights/768-v-ema.ckpt"

# Load config from path
print(f"Loading config from {config_path}")
config = OmegaConf.load(f"{config_path}")

# Load model from checkpoint file
print(f"Loading checkpoint from {checkpoint_path}")
pl_sd = torch.load(checkpoint_path, map_location="cpu")

# Extract state dictionary
sd = pl_sd["state_dict"]

# Instantiate model
model = instantiate_from_config(config.model)

# Load state dictionary
m, u = model.load_state_dict(sd, strict=False)

# Setup device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Send to device
model = model.to(device)

# Set sampler
sampler = DDIMSampler(model)

# Set prompt
prompt = "a python playing football with other rainforest animals"

# Condition
condition = model.get_learned_conditioning([prompt])

# Sample
steps = 50
n_samples = 1
shape = [4, 768 // 8 , 768 // 8]
sample, _ = sampler.sample(S=steps,
                                    conditioning=condition,
                                    batch_size=1,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=1.0,
                                    unconditional_conditioning=None,
                                    eta=0.0,
                                    x_T=None)


#FIN