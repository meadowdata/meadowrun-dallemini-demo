import os

import boto3
import wandb

import urllib.request

from linux.dalle_consts import *


def download_pretrained_dallemini_cache_in_s3(model_version_str: str, s3_bucket: str, s3_bucket_region: str) -> None:
    wandb.init(anonymous="must")

    model_version = ModelSize[model_version_str.upper()]
    if model_version == ModelSize.MEGA_FULL:
        dalle_model = DALLE_MODEL_MEGA_FULL
    elif model_version == ModelSize.MEGA:
        dalle_model = DALLE_MODEL_MEGA
    else:
        dalle_model = DALLE_MODEL_MINI

    tmp_dir = "dalle_pretrained_model"

    artifact = wandb.Api().artifact(dalle_model)
    artifact.download(tmp_dir)

    s3 = boto3.client("s3", region_name=s3_bucket_region)
    for file in os.listdir(tmp_dir):
        s3.upload_file(os.path.join(tmp_dir, file), s3_bucket, f"{model_version}/{file}")


def download_pretrained_dallemini_from_s3(model_version: str, s3_bucket: str, s3_bucket_region: str) -> str:
    s3 = boto3.client("s3", region_name=s3_bucket_region)
    local_dir = os.path.join("/var/meadowrun/machine_cache", model_version)
    os.makedirs(local_dir, exist_ok=True)
    for file in s3.list_objects(Bucket=s3_bucket)["Contents"]:
        local_path = os.path.join("/var/meadowrun/machine_cache", file["Key"])
        if file["Key"].startswith(f"{model_version}/") and not os.path.exists(local_path):
            print(f"Downloading {file['Key']}")
            s3.download_file(s3_bucket, file["Key"], local_path)

    return local_dir


_CLIP_DIR = "clip"
_CLIP_FILE = "ViT-L-14.pt"
_GLID_3_XL_DIR = "glid3xl"
_GLID_3_XL_FILES = ["bert.pt", "kl-f8.pt", "finetune.pt"]


def download_pretrained_gild3xl_cache_in_s3(s3_bucket: str, s3_bucket_region: str):

    s3 = boto3.client("s3", region_name=s3_bucket_region)

    urllib.request.urlretrieve(f"https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/{_CLIP_FILE}", _CLIP_FILE)
    s3.upload_file(_CLIP_FILE, s3_bucket, f"{_CLIP_DIR}/{_CLIP_FILE}")

    for file_name in _GLID_3_XL_FILES:
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', 'meadowrun-demo')
        opener.retrieve(f"https://dall-3.com/models/glid-3-xl/{file_name}", file_name)
        s3.upload_file(file_name, s3_bucket, f"{_GLID_3_XL_DIR}/{file_name}")


def download_pretrained_glid3xl_from_s3(s3_bucket: str, s3_bucket_region: str):
    s3 = boto3.client("s3", region_name=s3_bucket_region)

    clip_local_dir = os.path.join("/var/meadowrun/machine_cache", _CLIP_DIR)
    os.makedirs(clip_local_dir, exist_ok=True)
    clip_local_path = os.path.join(clip_local_dir, _CLIP_FILE)
    if not os.path.exists(clip_local_path):
        s3.download_file(s3_bucket, f"{_CLIP_DIR}/{_CLIP_FILE}", clip_local_path)

    glid3xl_local_dir = os.path.join("/var/meadowrun/machine_cache", _GLID_3_XL_DIR)
    os.makedirs(glid3xl_local_dir, exist_ok=True)
    for file_name in _GLID_3_XL_FILES:
        local_path = os.path.join(glid3xl_local_dir, file_name)
        if not os.path.exists(local_path):
            s3.download_file(s3_bucket, f"{_GLID_3_XL_DIR}/{file_name}", local_path)

    return clip_local_path, glid3xl_local_dir


_SWINIR_DIR = "swinir"
_SWINIR_FILE = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"


def download_pretrained_swinir_cache_in_s3(s3_bucket: str, s3_bucket_region: str):
    s3 = boto3.client("s3", region_name=s3_bucket_region)

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'meadowrun-demo')]
    with opener.open(f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{_SWINIR_FILE}") as f:
        s3.upload_fileobj(f, s3_bucket, f"{_SWINIR_DIR}/{_SWINIR_FILE}")


def download_pretrained_swinir_from_s3(s3_bucket: str, s3_bucket_region: str):
    s3 = boto3.client("s3", region_name=s3_bucket_region)

    swinir_local_dir = os.path.join("/var/meadowrun/machine_cache", _SWINIR_DIR)
    os.makedirs(swinir_local_dir, exist_ok=True)
    local_path = os.path.join(swinir_local_dir, _SWINIR_FILE)
    if not os.path.exists(local_path):
        s3.download_file(s3_bucket, f"{_SWINIR_DIR}/{_SWINIR_FILE}", local_path)

    return local_path