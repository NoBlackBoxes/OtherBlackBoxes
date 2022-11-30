import torch
from omegaconf import OmegaConf # hmmmm

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

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    print(module)
    if reload:
        print("reloading")
#        module_imp = importlib.import_module(module)
#        importlib.reload(module_imp)
    return None#getattr(importlib.import_module(module, package=None), cls)

model = get_obj_from_str(config["target"])(**config.get("params", dict()))



module, cls = string.rsplit(".", 1)


if reload:
    module_imp = importlib.import_module(module)
    importlib.reload(module_imp)
return getattr(importlib.import_module(module, package=None), cls)



model = instantiate_from_config(config.model)
m, u = model.load_state_dict(sd, strict=False)
if len(m) > 0 and verbose:
    print("missing keys:")
    print(m)
if len(u) > 0 and verbose:
    print("unexpected keys:")
    print(u)

model.cuda()
model.eval()
return model



model_path = 
model = TheModelClass.load_from_checkpoint(ckpt_file_path)