"""
Evaluation Suite for All Experimental Arms (Phase 4)
Computes Image Quality (PSNR, SSIM, CIEDE2000, MSE) and Open-Set Siamese Linkage Privacy
for All Controlled Baselines, Ablations, and Pareto Frontier Arms (Reviewer M7 Deliverable).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_dir = os.path.dirname(_this_dir)
_root_dir = os.path.dirname(_proj_dir)
for p in [_this_dir, _proj_dir, _root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import csv
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.color import rgb2lab, deltaE_ciede2000

from models.fs_pfan import FSPFAN
from privacy_attacker import SiameseLinkageAttacker
from linkage_dataset import SurgicalLinkageDataset


# --- Fast Vectorized Privacy Metrics ---
def fast_auc(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def fast_acc(y_true, y_scores):
    preds = (np.asarray(y_scores) >= 0.5).astype(int)
    return float(np.mean(preds == np.asarray(y_true)))


def bootstrap_ci(y_true, y_pred_prob, n_bootstraps=1000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    auc_scores = []
    acc_scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        sub_t = y_true[idx]
        sub_p = y_pred_prob[idx]
        if len(np.unique(sub_t)) < 2:
            continue
        auc_scores.append(fast_auc(sub_t, sub_p))
        acc_scores.append(fast_acc(sub_t, sub_p))

    alpha = (100 - ci) / 2.0
    auc_low = np.percentile(auc_scores, alpha)
    auc_high = np.percentile(auc_scores, 100 - alpha)
    acc_low = np.percentile(acc_scores, alpha)
    acc_high = np.percentile(acc_scores, 100 - alpha)
    return (auc_low, auc_high), (acc_low, acc_high)


def compute_image_metrics(clean_np, restored_np):
    """Computes MSE, PSNR, SSIM, CIEDE2000 on uint8 RGB numpy arrays."""
    mse = float(np.mean((clean_np.astype(np.float32) - restored_np.astype(np.float32)) ** 2))
    psnr = float(psnr_fn(clean_np, restored_np, data_range=255))
    ssim = float(ssim_fn(clean_np, restored_np, channel_axis=2, data_range=255))
    lab1 = rgb2lab(clean_np)
    lab2 = rgb2lab(restored_np)
    ciede = float(np.mean(deltaE_ciede2000(lab1, lab2)))
    return mse, psnr, ssim, ciede


def evaluate_privacy_linkage(attacker_model, condition_dir, num_pairs=1000, batch_size=32, device="cuda", seed=42):
    ds = SurgicalLinkageDataset(
        split="test",
        condition=condition_dir,
        num_pairs=num_pairs,
        seed=seed
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    attacker_model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch["img1"].to(device)
            x2 = batch["img2"].to(device)
            labels = batch["label"].to(device)
            logits = attacker_model(x1, x2)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    acc = fast_acc(all_targets, all_preds)
    auc = fast_auc(all_targets, all_preds)
    (auc_low, auc_high), (acc_low, acc_high) = bootstrap_ci(all_targets, all_preds, seed=seed)
    return {
        "acc": acc,
        "acc_low": acc_low,
        "acc_high": acc_high,
        "auc": auc,
        "auc_low": auc_low,
        "auc_high": auc_high
    }


def generate_restored_images_for_model(checkpoint_path,
                                       output_dir,
                                       test_dir="project_files/datasets/disjoint_splits/test",
                                       device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = FSPFAN(num_train_classes=14).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".bmp")])
    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    with torch.no_grad():
        for fname in test_files:
            p = os.path.join(test_dir, fname)
            im = Image.open(p).convert("RGB")
            w, h = im.size
            smoky = im.crop((0, 0, w // 2, h))

            t_inp = to_tensor(smoky).unsqueeze(0).to(device)
            t_out = model(t_inp, return_adv=False)

            # Denormalize [-1, 1] -> [0, 255] uint8
            out_np = t_out.squeeze(0).cpu().clamp(-1.0, 1.0).numpy()
            out_np = ((out_np + 1.0) / 2.0 * 255.0).astype(np.uint8).transpose(1, 2, 0)

            out_im = Image.fromarray(out_np)
            out_im.save(os.path.join(output_dir, fname))


def generate_noise_baseline(base_dir, output_dir, noise_type="gaussian", scale=8.0):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    files = sorted([f for f in os.listdir(base_dir) if f.endswith(".bmp")])
    for fname in files:
        im = Image.open(os.path.join(base_dir, fname)).convert("RGB")
        arr = np.array(im, dtype=np.float32)
        if noise_type == "gaussian":
            noise = rng.normal(0, scale, arr.shape)
        elif noise_type == "laplace":
            noise = rng.laplace(0, scale, arr.shape)
        else:
            noise = 0
        perturbed = np.clip(arr + noise, 0, 255).astype(np.uint8)
        Image.fromarray(perturbed).save(os.path.join(output_dir, fname))


def evaluate_arm(arm_name,
                 image_dir,
                 test_dir,
                 attacker_model,
                 device="cuda"):
    print(f"Evaluating {arm_name} from {image_dir}...")
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".bmp")])

    psnrs, ssims, ciedes, mses = [], [], [], []
    for fname in test_files:
        p_comp = os.path.join(test_dir, fname)
        im = Image.open(p_comp).convert("RGB")
        w, h = im.size
        clean_gt = im.crop((w // 2, 0, w, h)).resize((256, 256), Image.Resampling.BILINEAR)
        clean_np = np.array(clean_gt)

        p_res = os.path.join(image_dir, fname)
        if not os.path.exists(p_res):
            raise FileNotFoundError(f"Missing restored image {p_res}")
        res_np = np.array(Image.open(p_res).convert("RGB"))

        mse, psnr, ssim, ciede = compute_image_metrics(clean_np, res_np)
        mses.append(mse)
        psnrs.append(psnr)
        ssims.append(ssim)
        ciedes.append(ciede)

    # Privacy linkage evaluation
    priv = evaluate_privacy_linkage(attacker_model, image_dir, device=device)

    return {
        "arm_name": arm_name,
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "ciede_mean": float(np.mean(ciedes)),
        "ciede_std": float(np.std(ciedes)),
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
        "acc": priv["acc"],
        "acc_low": priv["acc_low"],
        "acc_high": priv["acc_high"],
        "auc": priv["auc"],
        "auc_low": priv["auc_low"],
        "auc_high": priv["auc_high"]
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("========================================================")
    print("Phase 4: Evaluating All Experimental Arms & Ablations")
    print("========================================================")

    attacker_path = os.path.join(_proj_dir, "checkpoints", "privacy_attacker", "best_linkage_attacker.pth")
    attacker = SiameseLinkageAttacker(pretrained=False).to(device)
    ckpt_att = torch.load(attacker_path, map_location=device)
    attacker.load_state_dict(ckpt_att["model_state_dict"] if "model_state_dict" in ckpt_att else ckpt_att)
    attacker.eval()
    print("Siamese linkage attacker loaded successfully.")

    test_dir = os.path.join(_proj_dir, "datasets", "disjoint_splits", "test")
    pfan_test_dir = os.path.join(_proj_dir, "results", "pfan_desmoked_test")
    arms_base_dir = os.path.join(_proj_dir, "results", "arms")
    os.makedirs(arms_base_dir, exist_ok=True)

    # 1. Baseline noise perturbations
    gauss8_dir = os.path.join(arms_base_dir, "gaussian_sigma8")
    gauss16_dir = os.path.join(arms_base_dir, "gaussian_sigma16")
    lap8_dir = os.path.join(arms_base_dir, "laplace_b8")

    generate_noise_baseline(pfan_test_dir, gauss8_dir, noise_type="gaussian", scale=8.0)
    generate_noise_baseline(pfan_test_dir, gauss16_dir, noise_type="gaussian", scale=16.0)
    generate_noise_baseline(pfan_test_dir, lap8_dir, noise_type="laplace", scale=8.0)

    # 2. Model arms definitions
    model_arms = [
        {"name": "img_only", "ckpt": os.path.join(_proj_dir, "checkpoints", "fs_pfan", "img_only_best.pth")},
        {"name": "mbi_only", "ckpt": os.path.join(_proj_dir, "checkpoints", "fs_pfan", "mbi_only_best.pth")},
        {"name": "lat_only", "ckpt": os.path.join(_proj_dir, "checkpoints", "fs_pfan", "lat_only_best.pth")},
        {"name": "joint_fs_pfan", "ckpt": os.path.join(_proj_dir, "checkpoints", "fs_pfan", "joint_fs_pfan_best.pth")},
        {"name": "pareto_0.01", "ckpt": os.path.join(_proj_dir, "checkpoints", "fs_pfan", "pareto_0.01_best.pth")},
        {"name": "pareto_0.10", "ckpt": os.path.join(_proj_dir, "checkpoints", "fs_pfan", "pareto_0.10_best.pth")},
        {"name": "pareto_0.20", "ckpt": os.path.join(_proj_dir, "checkpoints", "fs_pfan", "pareto_0.20_best.pth")},
    ]

    for m in model_arms:
        out_d = os.path.join(arms_base_dir, m["name"])
        if os.path.exists(m["ckpt"]):
            print(f"Generating test images for {m['name']} from {m['ckpt']}...")
            generate_restored_images_for_model(m["ckpt"], out_d, test_dir=test_dir, device=device)

    # List of all arms to evaluate for Table III
    table3_arms = [
        {"name": "PFAN Baseline (No Defense)", "dir": pfan_test_dir, "cat": "Baseline"},
        {"name": "Gaussian Noise (sigma=8)", "dir": gauss8_dir, "cat": "Noise Perturbation"},
        {"name": "Gaussian Noise (sigma=16)", "dir": gauss16_dir, "cat": "Noise Perturbation"},
        {"name": "Laplace DP Noise (b=8)", "dir": lap8_dir, "cat": "Noise Perturbation"},
        {"name": "Output Image Adversary", "dir": os.path.join(arms_base_dir, "img_only"), "cat": "Adversarial Baseline"},
        {"name": "MBI-Only (High-Freq Defense)", "dir": os.path.join(arms_base_dir, "mbi_only"), "cat": "Ablation"},
        {"name": "LAT-Only (Low-Freq Defense)", "dir": os.path.join(arms_base_dir, "lat_only"), "cat": "Ablation"},
        {"name": "FS-PFAN Full Joint Defense", "dir": os.path.join(arms_base_dir, "joint_fs_pfan"), "cat": "Proposed"}
    ]

    table3_results = []
    for arm in table3_arms:
        res = evaluate_arm(arm["name"], arm["dir"], test_dir, attacker, device=device)
        res["category"] = arm["cat"]
        table3_results.append(res)

    # Save Table III CSV
    t3_csv_path = os.path.join(_proj_dir, "results", "table3_baselines_ablations.csv")
    with open(t3_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "arm_name", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
            "ciede_mean", "ciede_std", "mse_mean", "mse_std",
            "acc", "acc_low", "acc_high", "auc", "auc_low", "auc_high"
        ])
        writer.writeheader()
        writer.writerows(table3_results)

    # Save Table III Markdown
    t3_md_path = os.path.join(_proj_dir, "results", "table3_baselines_ablations.md")
    with open(t3_md_path, "w", encoding="utf-8") as f:
        f.write("# Table III: Controlled Baselines & Architectural Ablations\n\n")
        f.write("| Category | Method / Defense | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | MSE ↓ | Attack Acc (%) [95% CI] ↓ | Attack AUC [95% CI] ↓ |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in table3_results:
            psnr_str = f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}"
            ssim_str = f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}"
            ciede_str = f"{r['ciede_mean']:.2f} ± {r['ciede_std']:.2f}"
            mse_str = f"{r['mse_mean']:.1f} ± {r['mse_std']:.1f}"
            acc_str = f"{r['acc']*100:.1f}% [{r['acc_low']*100:.1f}%, {r['acc_high']*100:.1f}%]"
            auc_str = f"{r['auc']:.4f} [{r['auc_low']:.4f}, {r['auc_high']:.4f}]"
            f.write(f"| {r['category']} | {r['arm_name']} | {psnr_str} | {ssim_str} | {ciede_str} | {mse_str} | {acc_str} | {auc_str} |\n")

    print(f"Table III saved to {t3_csv_path} and {t3_md_path}")

    # Evaluate Pareto Frontier (Table IV)
    table4_arms = [
        {"name": "lambda = 0.00 (PFAN Baseline)", "dir": pfan_test_dir, "lambda": "0.00"},
        {"name": "lambda = 0.01 (Low Privacy)", "dir": os.path.join(arms_base_dir, "pareto_0.01"), "lambda": "0.01"},
        {"name": "lambda = 0.05 (Default Joint FS-PFAN)", "dir": os.path.join(arms_base_dir, "joint_fs_pfan"), "lambda": "0.05"},
        {"name": "lambda = 0.10 (Moderate Privacy)", "dir": os.path.join(arms_base_dir, "pareto_0.10"), "lambda": "0.10"},
        {"name": "lambda = 0.20 (High Privacy)", "dir": os.path.join(arms_base_dir, "pareto_0.20"), "lambda": "0.20"}
    ]

    table4_results = []
    for arm in table4_arms:
        res = evaluate_arm(arm["name"], arm["dir"], test_dir, attacker, device=device)
        res["lambda"] = arm["lambda"]
        table4_results.append(res)

    # Save Table IV CSV
    t4_csv_path = os.path.join(_proj_dir, "results", "table4_pareto_frontier.csv")
    with open(t4_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "lambda", "arm_name", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
            "ciede_mean", "ciede_std", "mse_mean", "mse_std",
            "acc", "acc_low", "acc_high", "auc", "auc_low", "auc_high"
        ])
        writer.writeheader()
        writer.writerows(table4_results)

    # Save Table IV Markdown
    t4_md_path = os.path.join(_proj_dir, "results", "table4_pareto_frontier.md")
    with open(t4_md_path, "w", encoding="utf-8") as f:
        f.write("# Table IV: Privacy-Utility Tradeoff Pareto Frontier (Varying lambda)\n\n")
        f.write("| Tradeoff Arm (lambda) | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | Attack Acc (%) [95% CI] ↓ | Attack AUC [95% CI] ↓ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in table4_results:
            psnr_str = f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}"
            ssim_str = f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}"
            ciede_str = f"{r['ciede_mean']:.2f} ± {r['ciede_std']:.2f}"
            acc_str = f"{r['acc']*100:.1f}% [{r['acc_low']*100:.1f}%, {r['acc_high']*100:.1f}%]"
            auc_str = f"{r['auc']:.4f} [{r['auc_low']:.4f}, {r['auc_high']:.4f}]"
            f.write(f"| {r['arm_name']} | {psnr_str} | {ssim_str} | {ciede_str} | {acc_str} | {auc_str} |\n")

    print(f"Table IV saved to {t4_csv_path} and {t4_md_path}")
    print("\nPhase 4 Evaluation Complete!")


if __name__ == "__main__":
    main()

