# https://github.com/JingyunLiang/SwinIR/blob/main/main_test_swinir.py

import numpy as np
from collections import OrderedDict
import torch

from models.network_swinir import SwinIR as net

from linux.cache_in_s3 import download_pretrained_swinir_from_s3


def main(input_image, s3_bucket, s3_bucket_region):
    # TASK = "real_sr"
    SCALE = 4
    MODEL_PATH = download_pretrained_swinir_from_s3(s3_bucket, s3_bucket_region)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(SCALE, MODEL_PATH)
    model.eval()
    model = model.to(device)

    # setup folder and path
    window_size = 8
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []

    img_lq = input_image.astype(np.float32) / 255.0
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = test(img_lq, model)
        output = output[..., :h_old * SCALE, :w_old * SCALE]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    return output


def define_model(scale, model_path):
    # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
    model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key_g = 'params_ema'

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def test(img_lq, model):
    # test the image as a whole
    output = model(img_lq)

    return output
