# AI : ONNX : amd

## Archlinux
- Install rocminfo (binary located in /opt/rocm/bin)

```bash
rocminfo | grep Name
export AMDGPU_TARGETS="gfx####
```

- Install ROCm using "paru"
  - Add the arch4edu repository to pacman.conf (https://github.com/arch4edu/arch4edu/wiki/Add-arch4edu-to-your-Archlinux)

```bash
paru -S rocm-hip-sdk rocm-opencl-sdk
```

- Build ONNX runtime (with ROCm version 5.4)

```bash
cd ../tmp
git clone https://github.com/microsoft/onnxruntime

# This direct build did not work...
export CMAKE_PREFIX_PATH="/opt/rocm/lib/cmake/"
./build.sh --config=Release --use_rocm --rocm_home=/opt/rocm --build_wheel

# Docker (neother did this...)
cd onnxruntime/dockerfiles
docker build -t onnxruntime-rocm -f Dockerfile.rocm .
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video onnxruntime-rocm
```