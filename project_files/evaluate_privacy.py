"""
Evaluate Privacy Linkage Attacker (H1 Baseline Results)
Part of Phase 1: Privacy Attack Infrastructure & Audit.

Addresses Reviewer Feedback M7:
- Evaluates the trained Siamese Linkage Attacker on held-out test clusters across:
  1. Smoky degraded input frames
  2. Clean ground truth frames
  3. PFAN Desmoked restored frames
- Computes Linkage Accuracy, Linkage AUC-ROC, and 95% Confidence Intervals (Bootstrap).
- Outputs the canonical H1 results table.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from models.privacy_attacker import SiameseLinkageAttacker
from data.linkage_dataset import SurgicalLinkageDataset



def bootstrap_ci(y_true, y_pred_prob, n_bootstraps=1000, ci=95, seed=42):
    """Compute empirical bootstrap confidence intervals for AUC and Accuracy."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)

    auc_scores = []
    acc_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        # Ensure at least two classes present in sample
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc_scores.append(roc_auc_score(y_true[indices], y_pred_prob[indices]))
        acc_scores.append(accuracy_score(y_true[indices], y_pred_bin[indices]))

    alpha = (100 - ci) / 2.0
    auc_low = np.percentile(auc_scores, alpha)
    auc_high = np.percentile(auc_scores, 100 - alpha)
    acc_low = np.percentile(acc_scores, alpha)
    acc_high = np.percentile(acc_scores, 100 - alpha)

    return (auc_low, auc_high), (acc_low, acc_high)


def evaluate_condition(model, condition, num_pairs=1000, batch_size=32, device="cuda", seed=42):
    """Evaluate privacy linkage attack for a specific image condition on test split."""
    ds = SurgicalLinkageDataset(
        metadata_csv="project_files/metadata/frame_metadata.csv",
        split="test",
        condition=condition,
        num_pairs=num_pairs,
        seed=seed
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x1 = batch["img1"].to(device)
            x2 = batch["img2"].to(device)
            labels = batch["label"].to(device)

            logits = model(x1, x2)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, [1 if p >= 0.5 else 0 for p in all_preds])
    auc = roc_auc_score(all_targets, all_preds)
    (auc_low, auc_high), (acc_low, acc_high) = bootstrap_ci(all_targets, all_preds, seed=seed)

    return {
        "acc": acc,
        "acc_ci": (acc_low, acc_high),
        "auc": auc,
        "auc_ci": (auc_low, auc_high),
        "preds": all_preds,
        "targets": all_targets
    }


def run_h1_evaluation(checkpoint_path="project_files/checkpoints/privacy_attacker/best_linkage_attacker.pth",
                       desmoked_dir="project_files/results/pfan_desmoked_test",
                       output_dir="project_files/results",
                       num_pairs=1000,
                       seed=42):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running H1 Privacy Linkage Evaluation on {device}...")

    # Load trained linkage attacker
    model = SiameseLinkageAttacker(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded linkage attacker from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    conditions = [
        ("Smoky (Degraded Input)", "smoky"),
        ("Clean (Ground Truth)", "clean"),
        ("PFAN Desmoked (Restored Output)", desmoked_dir)
    ]

    records = []

    for label, cond in conditions:
        print(f"\nAuditing condition: {label}...")
        res = evaluate_condition(model, cond, num_pairs=num_pairs, device=device, seed=seed)
        acc_str = f"{res['acc']*100:.2f}% [{res['acc_ci'][0]*100:.2f}%, {res['acc_ci'][1]*100:.2f}%]"
        auc_str = f"{res['auc']:.4f} [{res['auc_ci'][0]:.4f}, {res['auc_ci'][1]:.4f}]"

        print(f"  Linkage Accuracy : {acc_str}")
        print(f"  Linkage AUC-ROC  : {auc_str}")

        records.append({
            "Frame Condition": label,
            "Linkage Accuracy (%)": round(res["acc"] * 100, 2),
            "Accuracy 95% CI": f"[{res['acc_ci'][0]*100:.2f}%, {res['acc_ci'][1]*100:.2f}%]",
            "Linkage AUC-ROC": round(res["auc"], 4),
            "AUC 95% CI": f"[{res['auc_ci'][0]:.4f}, {res['auc_ci'][1]:.4f}]"
        })

    # Add random chance baseline
    records.append({
        "Frame Condition": "Random Chance Baseline",
        "Linkage Accuracy (%)": 50.00,
        "Accuracy 95% CI": "—",
        "Linkage AUC-ROC": 0.5000,
        "AUC 95% CI": "—"
    })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "h1_baseline_results.csv")
    md_path = os.path.join(output_dir, "h1_baseline_results.md")

    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as fp:
        fp.write("# Table I: H1 Baseline Privacy Linkage Audit\n\n")
        fp.write(df.to_markdown(index=False))
        fp.write("\n")

    print("\n" + "="*70)
    print("=== TABLE I: H1 BASELINE PRIVACY LINKAGE AUDIT (Reviewer M7 Deliverable) ===")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    print(f"Saved results to {csv_path} and {md_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate H1 privacy leakage.")
    parser.add_argument("--checkpoint", default="project_files/checkpoints/privacy_attacker/best_linkage_attacker.pth")
    parser.add_argument("--desmoked_dir", default="project_files/results/pfan_desmoked_test")
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_h1_evaluation(
        checkpoint_path=args.checkpoint,
        desmoked_dir=args.desmoked_dir,
        num_pairs=args.num_pairs,
        seed=args.seed
    )
