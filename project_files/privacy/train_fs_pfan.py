"""
Training & Fine-tuning for FS-PFAN (Phase 3)
Frequency-Selective Privacy-Utility Optimization for Laparoscopic Image Desmoking.

Supports:
- Full Joint FS-PFAN (MBI + LAT adversaries)
- MBI-Only Frequency Adversary
- LAT-Only Frequency Adversary
- Image-Level Adversary Baseline
- Varying adversary regularization weights (Pareto Frontier)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_dir = os.path.dirname(_this_dir)
_root_dir = os.path.dirname(_proj_dir)
for p in [_this_dir, _proj_dir, _root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import csv
import math
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from models.fs_pfan import FSPFAN


# --- Pure PyTorch Differentiable SSIM Loss ---
def create_gaussian_window(window_size=11, sigma=1.5, channel=3):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w1d = g.unsqueeze(1)
    w2d = w1d.mm(w1d.t()).unsqueeze(0).unsqueeze(0)
    window = w2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_loss(img1, img2, window_size=11):
    """Computes (1.0 - SSIM) for inputs in range [-1, 1]."""
    # Shift from [-1, 1] to [0, 1]
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    channel = img1.size(1)
    window = create_gaussian_window(window_size=window_size, sigma=1.5, channel=channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return 1.0 - ssim_map.mean()


# --- Fast Metric Computation for Validation ---
def compute_psnr(pred, gt):
    """pred and gt in [-1, 1] float tensor."""
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return 100.0
    # Peak-to-peak range is 2.0
    return 10.0 * math.log10(4.0 / mse)


# --- Dataset Class for Disjoint Split Training ---
class DisjointSplitDataset(Dataset):
    def __init__(self, split_dir, metadata_csv, split_name="train", img_size=256):
        super().__init__()
        self.split_name = split_name
        self.split_dir = split_dir
        self.img_size = img_size

        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory {split_dir} does not exist.")

        self.files = sorted([f for f in os.listdir(split_dir) if f.endswith(".bmp")])
        if len(self.files) == 0:
            raise ValueError(f"No BMP files found in {split_dir}")

        # Load cluster map from metadata
        self.file_to_cluster = {}
        with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.file_to_cluster[row["bmp_name"]] = int(row["cluster_id"])

        # Train cluster list for index mapping: [0, 1, 2, 3, 5, 8, 9, 11, 13, 15, 16, 17, 18, 19]
        train_clusters = [0, 1, 2, 3, 5, 8, 9, 11, 13, 15, 16, 17, 18, 19]
        self.cluster_to_idx = {c: i for i, c in enumerate(train_clusters)}

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        print(f"Pre-caching {len(self.files)} frames in RAM for {split_name} split...")
        self.cached_data = []
        for fname in self.files:
            fpath = os.path.join(self.split_dir, fname)
            im = Image.open(fpath).convert("RGB")
            w, h = im.size
            smoky = im.crop((0, 0, w // 2, h)).resize((img_size, img_size), Image.Resampling.BILINEAR)
            clean = im.crop((w // 2, 0, w, h)).resize((img_size, img_size), Image.Resampling.BILINEAR)
            cluster_id = self.file_to_cluster.get(fname, 0)
            label = self.cluster_to_idx.get(cluster_id, 0)
            self.cached_data.append((smoky, clean, label, fname))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        smoky, clean, label, fname = self.cached_data[idx]

        if self.split_name == "train" and torch.rand(1).item() > 0.5:
            smoky = smoky.transpose(Image.FLIP_LEFT_RIGHT)
            clean = clean.transpose(Image.FLIP_LEFT_RIGHT)

        t_smoky = self.to_tensor_norm(smoky)
        t_clean = self.to_tensor_norm(clean)

        return {
            "smoky": t_smoky,
            "clean": t_clean,
            "label": torch.tensor(label, dtype=torch.long),
            "filename": fname
        }


def train_arm(arm_name,
              lambda_mbi=0.0,
              lambda_lat=0.0,
              lambda_img=0.0,
              epochs=6,
              batch_size=4,
              lr_g=1e-4,
              lr_adv=2e-4,
              pretrained_pfan="pfan_Final/latest_net_G.pth",
              save_dir="project_files/checkpoints/fs_pfan",
              device="cuda"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n=================================================================")
    print(f"Training Arm: {arm_name}")
    print(f"Lambdas: MBI={lambda_mbi}, LAT={lambda_lat}, IMG={lambda_img}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Device: {device}")
    print(f"=================================================================")

    # Initialize FS-PFAN model
    model = FSPFAN(num_train_classes=14).to(device)

    # Load pretrained PFAN weights
    if os.path.exists(pretrained_pfan):
        model.load_pfan_pretrained(pretrained_pfan)
    else:
        alt_path = os.path.join(_root_dir, pretrained_pfan)
        if os.path.exists(alt_path):
            model.load_pfan_pretrained(alt_path)
        else:
            raise FileNotFoundError(f"Pretrained PFAN checkpoint {pretrained_pfan} not found.")

    # Paths to datasets
    train_dir = os.path.join(_proj_dir, "datasets", "disjoint_splits", "train")
    val_dir = os.path.join(_proj_dir, "datasets", "disjoint_splits", "val")
    meta_csv = os.path.join(_proj_dir, "metadata", "frame_metadata.csv")

    train_ds = DisjointSplitDataset(train_dir, meta_csv, split_name="train")
    val_ds = DisjointSplitDataset(val_dir, meta_csv, split_name="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Parameter groups: generator backbone vs adversary heads
    backbone_params = [p for n, p in model.named_parameters() if "adv_" not in n and p.requires_grad]
    adv_params = [p for n, p in model.named_parameters() if "adv_" in n and p.requires_grad]

    optimizer_G = torch.optim.Adam(backbone_params, lr=lr_g, betas=(0.5, 0.999))
    optimizer_Adv = torch.optim.Adam(adv_params, lr=lr_adv, betas=(0.5, 0.999))

    scaler = torch.cuda.amp.GradScaler()

    criterion_ce = nn.CrossEntropyLoss()
    criterion_l1 = nn.L1Loss()

    best_val_score = -1e9
    best_checkpoint_path = os.path.join(save_dir, f"{arm_name}_best.pth")

    accum_steps = 2
    total_steps = (epochs * len(train_loader)) // accum_steps
    current_step = 0

    torch.cuda.empty_cache()

    for epoch in range(1, epochs + 1):
        model.train()
        train_l1_total = 0.0
        train_ssim_total = 0.0
        train_adv_total = 0.0
        t0 = time.time()

        optimizer_G.zero_grad()
        optimizer_Adv.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Dynamic GRL alpha schedule (Ganin et al. DANN)
            p_progress = min(1.0, float(current_step) / float(max(1, total_steps)))
            alpha = 2.0 / (1.0 + math.exp(-10.0 * p_progress)) - 1.0

            smoky = batch["smoky"].to(device)
            clean = batch["clean"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast():
                # Forward pass with adversaries
                out_img, preds = model(smoky, alpha=alpha, return_adv=True)

                # Restoration loss
                l1_loss = criterion_l1(out_img, clean)
                loss_ssim = ssim_loss(out_img, clean)
                rec_loss = l1_loss + 0.2 * loss_ssim

                # Adversary losses
                adv_loss = torch.tensor(0.0, device=device)
                if lambda_mbi > 0:
                    loss_mbi = criterion_ce(preds["mbi"], labels)
                    adv_loss = adv_loss + lambda_mbi * loss_mbi
                if lambda_lat > 0:
                    loss_lat = criterion_ce(preds["lat"], labels)
                    adv_loss = adv_loss + lambda_lat * loss_lat
                if lambda_img > 0:
                    loss_img = criterion_ce(preds["img"], labels)
                    adv_loss = adv_loss + lambda_img * loss_img

                total_loss = (rec_loss + adv_loss) / accum_steps

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer_G)
                scaler.step(optimizer_Adv)
                scaler.update()
                optimizer_G.zero_grad()
                optimizer_Adv.zero_grad()
                current_step += 1

            train_l1_total += l1_loss.item()
            train_ssim_total += (1.0 - loss_ssim.item())
            train_adv_total += adv_loss.item()

        n_batches = len(train_loader)
        avg_l1 = train_l1_total / n_batches
        avg_ssim = train_ssim_total / n_batches
        avg_adv = train_adv_total / n_batches

        # Evaluate on validation set
        model.eval()
        torch.cuda.empty_cache()
        val_psnr_total = 0.0
        val_ssim_total = 0.0
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for v_batch in val_loader:
                    v_smoky = v_batch["smoky"].to(device)
                    v_clean = v_batch["clean"].to(device)
                    v_out = model(v_smoky, return_adv=False)

                    val_psnr_total += compute_psnr(v_out.float(), v_clean.float())
                    val_ssim_total += (1.0 - ssim_loss(v_out.float(), v_clean.float()).item())

        val_batches = len(val_loader)
        mean_val_psnr = val_psnr_total / val_batches
        mean_val_ssim = val_ssim_total / val_batches

        # Combined validation score (utility focused)
        val_score = mean_val_psnr + 50.0 * mean_val_ssim
        epoch_time = time.time() - t0

        print(f"Epoch {epoch}/{epochs} [{epoch_time:.1f}s] - Train L1: {avg_l1:.4f}, SSIM: {avg_ssim:.4f}, AdvLoss: {avg_adv:.4f} | Val PSNR: {mean_val_psnr:.2f} dB, Val SSIM: {mean_val_ssim:.4f}")

        # Save best checkpoint
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save({
                "epoch": epoch,
                "arm_name": arm_name,
                "model_state_dict": model.state_dict(),
                "val_psnr": mean_val_psnr,
                "val_ssim": mean_val_ssim,
                "lambdas": {"mbi": lambda_mbi, "lat": lambda_lat, "img": lambda_img}
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path}")

    print(f"Arm {arm_name} training completed! Best checkpoint: {best_checkpoint_path}\n")
    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm_name", type=str, default="joint_fs_pfan")
    parser.add_argument("--lambda_mbi", type=float, default=0.025)
    parser.add_argument("--lambda_lat", type=float, default=0.025)
    parser.add_argument("--lambda_img", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_adv", type=float, default=2e-4)
    args = parser.parse_args()

    train_arm(
        arm_name=args.arm_name,
        lambda_mbi=args.lambda_mbi,
        lambda_lat=args.lambda_lat,
        lambda_img=args.lambda_img,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_adv=args.lr_adv
    )
