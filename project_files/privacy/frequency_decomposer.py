"""
2D Discrete Wavelet Transform (Haar DWT) Frequency Decomposer
Phase 2: Frequency-Domain Privacy Analysis (H2)

Pure PyTorch implementation of 2D Haar Wavelet Transform (no external C-libraries).
Decomposes surgical images into 4 orthogonal frequency sub-bands:
- LL: Low-Frequency Approximation (macroscopic tissue color, ambient illumination)
- LH: Horizontal High-Frequency Detail (horizontal tool edges, anatomical boundaries)
- HL: Vertical High-Frequency Detail (vertical tissue contours, rising smoke traces)
- HH: Diagonal High-Frequency Detail (fine surgical texture, specular glints, microsurgical incisions)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


class HaarDWT2D(nn.Module):
    """2D Haar Wavelet Decomposition via 2D Conv with fixed orthogonal kernels."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        self.register_buffer("ll", ll.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer("lh", lh.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer("hl", hl.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer("hh", hh.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))

    def forward(self, x):
        """
        Input x: Tensor of shape (B, 3, H, W)
        Output: Dictionary with keys 'LL', 'LH', 'HL', 'HH', each of shape (B, 3, H//2, W//2)
        """
        return {
            "LL": F.conv2d(x, self.ll, stride=2, groups=3),
            "LH": F.conv2d(x, self.lh, stride=2, groups=3),
            "HL": F.conv2d(x, self.hl, stride=2, groups=3),
            "HH": F.conv2d(x, self.hh, stride=2, groups=3)
        }



def decompose_and_save_dataset(desmoked_dir="project_files/results/pfan_desmoked_test",
                               output_base="project_files/results/frequency_subbands"):
    """
    Decomposes all PFAN desmoked test frames into LL, LH, HL, and HH sub-bands,
    resizes sub-bands back to 256x256 for consistent attacker evaluation,
    and saves them to designated folders.
    """
    subbands = ["LL", "LH", "HL", "HH"]
    for sb in subbands:
        os.makedirs(os.path.join(output_base, sb.lower()), exist_ok=True)

    dwt = HaarDWT2D().eval()
    files = sorted([f for f in os.listdir(desmoked_dir) if f.endswith(".bmp")])
    print(f"Decomposing {len(files)} PFAN desmoked images into frequency sub-bands (LL, LH, HL, HH)...", flush=True)

    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    with torch.no_grad():
        for f in files:
            img_path = os.path.join(desmoked_dir, f)
            img = Image.open(img_path).convert("RGB")
            t_img = to_tensor(img).unsqueeze(0)  # (1, 3, 256, 256)

            bands = dwt(t_img)

            for sb in subbands:
                sb_tensor = bands[sb].squeeze(0)  # (3, 128, 128)

                # For detail bands (LH, HL, HH), normalize signed wavelets to [0, 1] for visualization/eval
                if sb == "LL":
                    norm_sb = sb_tensor.clamp(0.0, 1.0)
                else:
                    norm_sb = (sb_tensor + 0.5).clamp(0.0, 1.0)

                # Resize back to (256, 256) so attacker can process identically
                norm_sb = F.interpolate(norm_sb.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0)

                np_img = (norm_sb.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                out_path = os.path.join(output_base, sb.lower(), f)
                Image.fromarray(np_img).save(out_path)

    print("Frequency decomposition complete! Sub-band folders created:", flush=True)
    for sb in subbands:
        p = os.path.join(output_base, sb.lower())
        print(f"  {sb:<4}: {len(os.listdir(p))} images in {p}", flush=True)


if __name__ == "__main__":
    decompose_and_save_dataset()
