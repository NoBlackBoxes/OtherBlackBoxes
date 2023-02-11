# Cloud : AWS

Using the cloud for compute and storage

## Setting up an EC2 instance for GPU compute

- Request a vCPU limit increase (seems to take awhile). Requires 8 vCPUs for a p3.2x

- Select an arch distribution for your region (eu-west-2 is London)
  - https://wiki.archlinux.org/title/Arch_Linux_AMIs_for_Amazon_Web_Services
  - There are optimized builds for EC2

- Wait for limit increase

## Setting up Google Compute Engine

- Activate Google Cloud account
- Enable the Compute Engine API
- Create instance (may have to raise limit)

```bash
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
    --local-ssd=interface=NVME \
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