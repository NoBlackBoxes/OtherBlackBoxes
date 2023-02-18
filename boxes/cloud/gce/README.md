# Cloud : GCE (Google Compute Engine)

Using the cloud for compute and storage (Google Cloud)

## Setting up Google Compute Engine

- Activate Google Cloud account
- Enable the Compute Engine API
- Create instance (may have to raise limit)

```bash
# GPU Simple (T4)
gcloud compute instances create vk-0005  \
    --project=vk-cloud-377513 \
    --zone=europe-west1-d \
    --machine-type=n1-standard-4 \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=614909497347-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-tesla-t4 \
    --create-disk=auto-delete=yes,boot=yes,device-name=vk-0005,image=projects/ml-images/global/images/c2-deeplearning-pytorch-1-13-cu113-v20230126-debian-10,mode=rw,size=128,type=projects/vk-cloud-377513/zones/europe-west1-b/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --reservation-affinity=any

# GPU Advanced (A100)
gcloud compute instances create vk-0005 \
    --project=vk-cloud-377513 \
    --zone=europe-west4-a \
    --machine-type=a2-highgpu-1g \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=614909497347-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-tesla-a100 \
    --create-disk=auto-delete=yes,boot=yes,device-name=vk-0005,image=projects/ml-images/global/images/c2-deeplearning-pytorch-1-13-cu113-v20230126-debian-10,mode=rw,size=128,type=projects/vk-cloud-377513/zones/europe-west4-a/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --reservation-affinity=any
```

- Generate SSH key

```bash
ssh-keygen -t rsa -f ~/.ssh/gce_rsa -C <USERNAME> -b 2048
```

- Add public key to project Metadata (found in gce_rsa.pub): ssh-ras <key> <USERNAME>
- Connect

```bash
ssh -i <keypath> <username>@<IP>
```

- Download datasets, clone repos, etc.

```bash
mkdir -p ~/NoBlackBoxes/repos
cd ~/NoBlackBoxes/repos
git clone https://github.com/NoBlackBoxes/OtherBlackBoxes
mkdir -p ~/Datasets/coco
cd ~/Datasets/coco
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm -rf train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm -rf annotations_trainval2017.zip
```

- Install Python packages

```bash
pip install opencv-python pycocotools torchsummary timm
```