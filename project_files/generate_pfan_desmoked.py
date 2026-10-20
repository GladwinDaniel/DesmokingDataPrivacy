"""
Generate PFAN Desmoked Test Frames and Evaluate Utility Metrics
Part of Phase 1: Privacy Attack Infrastructure & Baseline Evaluation.

Loads the 200-epoch pretrained PFAN generator (pfan_Final/latest_net_G.pth),
runs inference on all held-out test frames (datasets/disjoint_splits/test),
saves the desmoked outputs to results/pfan_desmoked_test,
and computes PSNR, SSIM, CIEDE2000, and MSE on the disjoint test set.
"""

import os
import sys
import functools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.color import rgb2lab, deltaE_ciede2000

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.pfan import PFAN


def compute_metrics(real_clean_np, desmoked_np):
    """
    Compute MSE, PSNR, SSIM, and CIEDE2000 between ground truth clean and desmoked.
    Inputs are uint8 numpy arrays of shape (H, W, 3).
    """
    # MSE
    mse = np.mean((real_clean_np.astype(np.float32) - desmoked_np.astype(np.float32)) ** 2)

    # PSNR
    psnr = psnr_fn(real_clean_np, desmoked_np, data_range=255)

    # SSIM
    ssim = ssim_fn(real_clean_np, desmoked_np, channel_axis=2, data_range=255)

    # CIEDE2000
    lab1 = rgb2lab(real_clean_np)
    lab2 = rgb2lab(desmoked_np)
    ciede = np.mean(deltaE_ciede2000(lab1, lab2))

    return mse, psnr, ssim, ciede


def generate_and_evaluate(checkpoint_path="pfan_Final/latest_net_G.pth",
                          test_dir="project_files/datasets/disjoint_splits/test",
                          output_dir="project_files/results/pfan_desmoked_test",
                          device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading pretrained PFAN from {checkpoint_path} on {device}...")

    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    net_G = PFAN(input_nc=3, output_nc=3, ngf=64, hidden_dim=64, layers=[2, 2, 2], heads=[4, 4, 4], norm_layer_1=norm_layer)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    net_G.load_state_dict(state_dict, strict=True)
    net_G.to(device).eval()
    print("PFAN generator successfully loaded (629K params).")

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".bmp")])
    print(f"Generating desmoked images for {len(test_files)} disjoint test frames...")

    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    results = []

    with torch.no_grad():
        for f in test_files:
            comp_path = os.path.join(test_dir, f)
            comp_img = Image.open(comp_path).convert("RGB")
            w, h = comp_img.size

            # Split composite into smoky (left) and clean ground truth (right)
            smoky_img = comp_img.crop((0, 0, w // 2, h))
            clean_gt_img = comp_img.crop((w // 2, 0, w, h))

            # Resize clean_gt to 256x256 for consistent metric comparison
            clean_gt_res = clean_gt_img.resize((256, 256), Image.Resampling.BILINEAR)
            clean_gt_np = np.array(clean_gt_res)

            # Preprocess smoky image
            inp_tensor = to_tensor(smoky_img).unsqueeze(0).to(device)

            # Generator inference
            pred_tensor = net_G(inp_tensor)

            # Denormalize from [-1, 1] to [0, 255] uint8
            pred_img_np = ((pred_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            # Save desmoked frame
            out_img = Image.fromarray(pred_img_np)
            out_img.save(os.path.join(output_dir, f))

            # Compute restoration quality metrics
            mse, psnr, ssim, ciede = compute_metrics(clean_gt_np, pred_img_np)
            results.append({
                "bmp_name": f,
                "MSE": mse,
                "PSNR": psnr,
                "SSIM": ssim,
                "CIEDE2000": ciede
            })

    df = pd.DataFrame(results)
    summary_csv = os.path.join("project_files/results", "pfan_baseline_test_metrics.csv")
    df.to_csv(summary_csv, index=False)

    print("\n=== PFAN Baseline Restoration Performance on Disjoint Test Split (N=103) ===")
    print(f"PSNR (dB)   : {df['PSNR'].mean():.2f} ± {df['PSNR'].std():.2f}")
    print(f"SSIM        : {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
    print(f"CIEDE2000   : {df['CIEDE2000'].mean():.2f} ± {df['CIEDE2000'].std():.2f}")
    print(f"MSE         : {df['MSE'].mean():.2f} ± {df['MSE'].std():.2f}")
    print(f"Saved desmoked frames to {output_dir}")
    print(f"Saved per-frame metrics to {summary_csv}")


if __name__ == "__main__":
    generate_and_evaluate()
